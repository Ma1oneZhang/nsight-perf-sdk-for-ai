//! `gt dump parquet|perfetto <start> <end>`: every persisted activity that starts in `[start, end)` (the daemon's
//! `persist.dir`, see [`crate::persist`]) into one trace Parquet file (+ PM samples) or one gzip-compressed Perfetto
//! JSON file, named `<start>-<end>.parquet` / `<start>-<end>.json.gz` (UTC ns).
//!
//! Reading and writing both go through the object store ([`crate::store`]): files are picked by the time span in
//! their names, row groups by their `start_ns` statistics (only the chosen ones are fetched), rows by `start_ns`. The
//! Perfetto converter reads the filtered batches directly (two passes over the same objects), with no intermediate
//! file. When the range reaches past the newest file and the daemon is serving, it is first asked to close its open
//! file (`POST /persist/flush`), so everything merged up to now is included (the last snapshot interval is not merged
//! yet).
use crate::config;
use crate::error::{BadTimeSnafu, CoreSnafu, EmptyRangeSnafu, JoinSnafu, NoDataSnafu, PerfettoFilesSnafu, Result};
use crate::files::{self, FileKind, StoredFile};
use crate::ranks::Ranks;
use crate::store::{Batches, Object, ObjectWriter, Scan, Store};
use crate::timefmt::{fmt_span, fmt_utc, now_ns, parse_time};
use crate::writer::{CHROME_EXT, PM_EXT, TRACE_EXT, Written, pm_schema, span_name, write_chrome_from};
use gravity_trace_core::chrome::{GPUS_META, PROCESSES_META};
use gravity_trace_core::model::ProcessInfo;
use gravity_trace_core::parquet_io::{RankFilter, TableWriter, WriteOptions, collect_rank_pids};
use gravity_trace_core::schema::trace_schema;
use rustc_hash::{FxHashMap, FxHashSet};
use snafu::{OptionExt, ResultExt, ensure};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::runtime::Handle;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    /// Trace Parquet (same layout as the persisted files) + PM samples (`.pm.parquet`).
    Parquet,
    /// Perfetto / Chrome trace JSON, gzip-compressed (`.json.gz`), PM samples as counter tracks; open in
    /// https://ui.perfetto.dev.
    Perfetto,
}

pub struct DumpArgs {
    pub format: Format,
    pub start: String,
    pub end: String,
    /// Persisted trace location (default: persist.dir of the configuration).
    pub dir: Option<String>,
    /// Output directory / object store URL, or the output file in it.
    pub out: String,
    /// Only these ranks (CUDA device ids, comma-separated) and their processes' API calls.
    pub rank: Option<String>,
    pub no_pm: bool,
    pub no_flush: bool,
    pub json: bool,
}

#[derive(Debug, serde::Serialize)]
pub struct DumpReport {
    pub format: Format,
    pub start_utc_ns: i64,
    pub end_utc_ns: i64,
    pub dir: String,
    /// The rank filter, when given (CUDA device ids).
    pub ranks: Option<Vec<u32>>,
    /// What the daemon answered to `POST /persist/flush`, or why it was not asked.
    pub flush: Option<String>,
    pub trace_files: usize,
    pub scan: Scan,
    pub output: String,
    pub output_bytes: u64,
    pub pm: Option<String>,
    pub pm_samples: u64,
    pub elapsed_s: f64,
}

pub async fn dump_cmd(a: DumpArgs, config_path: Option<&Path>) -> Result<()> {
    let t0 = Instant::now();
    let c = config::load(config_path)?.config;
    let dir = a.dir.clone().unwrap_or_else(|| c.persist.dir.clone());
    let now = now_ns();
    let start = parse_time(&a.start, now, None).context(BadTimeSnafu { input: a.start.clone() })?;
    let end = parse_time(&a.end, now, Some(start)).context(BadTimeSnafu { input: a.end.clone() })?;
    ensure!(start < end, EmptyRangeSnafu { start: fmt_utc(start), end: fmt_utc(end) });
    let ranks = a.rank.as_deref().map(Ranks::parse).transpose()?;
    if !a.json {
        let what = match &ranks {
            Some(r) => format!("ranks {r} "),
            None => String::new(),
        };
        eprintln!("dump {} .. {} ({}) {what}from {dir}", fmt_utc(start), fmt_utc(end), fmt_span(end - start));
    }

    let store = Store::open(&dir)?;
    let mut files = files::list(&store).await?;
    let newest = files.iter().filter(|f| f.kind == FileKind::Trace).map(|f| f.span.1).max().unwrap_or(0);
    let mut flush = None;
    if end > newest && !a.no_flush && same_location(&dir, &c.persist.dir) {
        flush = Some(ask_daemon_to_flush(&c).await);
        if let Some(msg) = &flush
            && !a.json
        {
            eprintln!("  {msg}");
        }
        files = files::list(&store).await?;
    }

    let pick = |kind: FileKind| -> Vec<Object> {
        files.iter().filter(|f| f.kind == kind && f.span.0 < end && f.span.1 >= start).map(StoredFile::object).collect()
    };
    let traces = pick(FileKind::Trace);
    // PM samples: a `.pm.parquet` next to a Parquet dump, counter tracks in a Perfetto one
    let pms = if a.no_pm { Vec::new() } else { pick(FileKind::Pm) };
    if traces.is_empty() && !pick(FileKind::Perfetto).is_empty() {
        return PerfettoFilesSnafu { dir }.fail();
    }
    if traces.is_empty() {
        return NoDataSnafu { dir, start: fmt_utc(start), end: fmt_utc(end), have: coverage(&files) }.fail();
    }
    let (out_loc, name) = output(a.format, &a.out, (start, end));
    let out = Store::open(&out_loc)?;
    if a.format == Format::Perfetto && !a.json {
        eprintln!("  converting to Perfetto JSON ...");
    }
    let n_traces = traces.len();
    let (format, rt) = (a.format, Handle::current());
    let src = store.clone();
    let rs = ranks.clone();
    let (scan, written, pm) =
        tokio::task::spawn_blocking(move || write(format, &src, &rt, &traces, &pms, (start, end), &out, &name, rs.as_ref()))
            .await
            .context(JoinSnafu)??;
    let Some(written) = written else {
        return NoDataSnafu { dir, start: fmt_utc(start), end: fmt_utc(end), have: coverage(&files) }.fail();
    };
    let r = DumpReport {
        format: a.format,
        start_utc_ns: start,
        end_utc_ns: end,
        dir,
        ranks: ranks.map(|r| r.devices().to_vec()),
        flush,
        trace_files: n_traces,
        output: written.shown,
        output_bytes: written.bytes,
        scan,
        pm_samples: pm.as_ref().map_or(0, |p| p.1),
        pm: pm.map(|p| p.0.shown),
        elapsed_s: t0.elapsed().as_secs_f64(),
    };
    if a.json {
        println!("{}", serde_json::to_string_pretty(&r).unwrap_or_default());
        return Ok(());
    }
    let s = &r.scan;
    eprintln!(
        "  {} trace files, {} row groups read ({} skipped by time): {} activities",
        r.trace_files, s.row_groups, s.row_groups_skipped, s.rows
    );
    println!("wrote {} ({:.1} MiB)", r.output, r.output_bytes as f64 / (1 << 20) as f64);
    if let Some(p) = &r.pm {
        println!("wrote {p} ({} PM samples)", r.pm_samples);
    }
    if a.format == Format::Perfetto {
        eprintln!("  open in https://ui.perfetto.dev");
    }
    eprintln!("done in {:.1}s", r.elapsed_s);
    Ok(())
}

/// (location, file name) of the output: `<out>/<start>-<end>.parquet` / `.json.gz`, or `out` itself when it names a
/// file (Perfetto output is always gzip).
fn output(format: Format, out: &str, range: (i64, i64)) -> (String, String) {
    let (dir, file) = match out.rsplit_once('/') {
        Some(("", f)) => ("/", f),
        Some((d, f)) => (d, f),
        None => (".", out),
    };
    match format {
        Format::Parquet if file.ends_with(".parquet") => (dir.to_string(), file.to_string()),
        Format::Perfetto if file.ends_with(".gz") => (dir.to_string(), file.to_string()),
        Format::Perfetto if file.ends_with(".json") => (dir.to_string(), format!("{file}.gz")),
        Format::Parquet => (out.to_string(), span_name(range, TRACE_EXT)),
        Format::Perfetto => (out.to_string(), span_name(range, CHROME_EXT)),
    }
}

/// What the persisted trace covers, for "no data" errors.
fn coverage(files: &[StoredFile]) -> String {
    let t: Vec<_> = files.iter().filter(|f| f.kind == FileKind::Trace).collect();
    match (t.iter().map(|f| f.span.0).min(), t.iter().map(|f| f.span.1).max()) {
        (Some(s), Some(e)) => format!("{} trace files cover {} .. {}", t.len(), fmt_utc(s), fmt_utc(e)),
        _ => "no trace files".to_string(),
    }
}

fn same_location(a: &str, b: &str) -> bool {
    if a.contains("://") || b.contains("://") {
        return a.trim_end_matches('/') == b.trim_end_matches('/');
    }
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => a == b,
    }
}

/// Ask a serving daemon to close its open file; a line for the report either way.
async fn ask_daemon_to_flush(c: &config::Config) -> String {
    if c.daemon.listen.is_empty() {
        return "the daemon has no HTTP address (daemon.listen); its open file is not included".to_string();
    }
    if crate::pidfile::running_pid(&c.daemon.pid_file).is_none() {
        return "no daemon running; dumping what is stored".to_string();
    }
    let addr = local_addr(&c.daemon.listen);
    match post(&addr, "/persist/flush").await {
        Ok(body) => {
            let v: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
            match v.get("trace").and_then(|t| t.as_str()) {
                Some(t) => format!(
                    "the daemon closed its open file: {} ({} activities)",
                    t.rsplit('/').next().unwrap_or(t),
                    v.get("rows").and_then(|r| r.as_u64()).unwrap_or(0)
                ),
                None => "the daemon had no open file".to_string(),
            }
        }
        Err(e) => format!("cannot ask the daemon at {addr} to close its open file ({e}); it is not included"),
    }
}

/// The address to connect to for a listen address (`0.0.0.0:9465` -> `127.0.0.1:9465`).
fn local_addr(listen: &str) -> String {
    if let Some(port) = listen.strip_prefix("0.0.0.0:") {
        format!("127.0.0.1:{port}")
    } else if let Some(port) = listen.strip_prefix("[::]:") {
        format!("[::1]:{port}")
    } else if listen.starts_with(':') {
        format!("127.0.0.1{listen}")
    } else {
        listen.to_string()
    }
}

/// A bodyless HTTP/1.1 POST; the response body on 200.
async fn post(addr: &str, path: &str) -> std::result::Result<String, String> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let exchange = async {
        let mut s = tokio::net::TcpStream::connect(addr).await?;
        let req = format!("POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        s.write_all(req.as_bytes()).await?;
        let mut buf = Vec::new();
        s.read_to_end(&mut buf).await?;
        Ok::<_, std::io::Error>(String::from_utf8_lossy(&buf).into_owned())
    };
    // Closing a file sorts and compresses up to one row group: seconds, not minutes.
    let resp = tokio::time::timeout(Duration::from_secs(120), exchange)
        .await
        .map_err(|_| "no answer in 120 s".to_string())?
        .map_err(|e| e.to_string())?;
    let (head, body) = resp.split_once("\r\n\r\n").unwrap_or((&resp, ""));
    let status = head.split_whitespace().nth(1).unwrap_or("");
    if status == "200" { Ok(body.to_string()) } else { Err(format!("{} {}", head.lines().next().unwrap_or(""), body.trim())) }
}

type Dumped = (Scan, Option<Written>, Option<(Written, u64)>);

/// What a dump with `--rank` keeps, learned from the picked trace files and their footers.
struct RankSelection {
    /// The trace rows (GPU rows of the devices, their processes' API calls).
    trace: RankFilter,
    /// The PM samples of the selected devices: uuid -> how its GPU is named in the output (`rank <device>`).
    /// None when no trace file carried the gpu -> device mapping (files from before it was recorded): the PM
    /// samples then pass through unchanged.
    pm: Option<FxHashMap<String, String>>,
}

/// The processes with GPU activity on the selected devices in `range` (their API calls, which carry no device,
/// belong to the dump too) and the GPUs' device ids from the files' footers — over a pid / device projection.
fn select_ranks(src: &Store, rt: &Handle, traces: &[Object], range: (i64, i64), ranks: &Ranks) -> Result<RankSelection> {
    let mut pids = FxHashSet::default();
    let mut dev_of: FxHashMap<String, u32> = FxHashMap::default();
    let mut b = Batches {
        store: src,
        rt,
        objects: traces,
        filter: Some(("start_ns", range.0, range.1)),
        schema: None,
        columns: Some(&["pid", "device_id", "start_ns"]),
        rank: None,
        pm_rank: None,
        scan: Scan::default(),
    };
    b.each(|path, meta, batch| {
        collect_rank_pids(&batch, ranks.devices(), &mut pids, path).context(CoreSnafu)?;
        if let Some(j) = meta.get(GPUS_META) {
            let list: Vec<(String, u32)> = serde_json::from_str(j).unwrap_or_default();
            dev_of.extend(list);
        }
        Ok(())
    })?;
    let trace = RankFilter { devices: ranks.devices().to_vec(), pids };
    let pm = match dev_of.is_empty() {
        true => None,
        false => Some(
            dev_of
                .into_iter()
                .filter(|(_, d)| ranks.devices().contains(d))
                .map(|(u, d)| (u, format!("rank {d}")))
                .collect(),
        ),
    };
    Ok(RankSelection { trace, pm })
}

/// Write the dump (on a blocking thread: the stores are driven through `rt`).
#[allow(clippy::too_many_arguments)]
fn write(
    format: Format,
    src: &Store,
    rt: &Handle,
    traces: &[Object],
    pms: &[Object],
    range: (i64, i64),
    out: &Store,
    name: &str,
    ranks: Option<&Ranks>,
) -> Result<Dumped> {
    let meta = BTreeMap::from([
        ("source".to_string(), "gt dump".to_string()),
        ("dump_start_utc_ns".to_string(), range.0.to_string()),
        ("dump_end_utc_ns".to_string(), range.1.to_string()),
        ("host".to_string(), crate::host()),
    ]);
    let sel = match ranks {
        Some(r) => Some(select_ranks(src, rt, traces, range, r)?),
        None => None,
    };
    if let Some(s) = &sel
        && s.pm.is_none()
        && !pms.is_empty()
    {
        eprintln!("gt dump: warning: the trace files carry no gpu -> device mapping (written before it was recorded): PM samples are kept, unfiltered");
    }
    let mut batches = Batches {
        store: src,
        rt,
        objects: traces,
        filter: Some(("start_ns", range.0, range.1)),
        schema: Some(trace_schema()),
        columns: None,
        rank: sel.as_ref().map(|s| &s.trace),
        pm_rank: None,
        scan: Scan::default(),
    };
    let written = match format {
        Format::Parquet => {
            let opts = WriteOptions { metadata: meta.clone(), ..WriteOptions::trace() };
            copy(&mut batches, out, rt, name, &opts)?
        }
        Format::Perfetto => {
            let mut pm_batches = Batches {
                store: src,
                rt,
                objects: pms,
                filter: Some(("end_ns", range.0, range.1)),
                schema: Some(pm_schema()),
                columns: None,
                rank: None,
                pm_rank: sel.as_ref().and_then(|s| s.pm.as_ref()),
                scan: Scan::default(),
            };
            let pm_opt: Option<&mut Batches<'_>> = if pms.is_empty() { None } else { Some(&mut pm_batches) };
            let w = write_chrome_from(out, rt, &mut batches, pm_opt, name)?;
            if batches.scan.rows == 0 {
                let _ = rt.block_on(out.delete(name));
                None
            } else {
                Some(w)
            }
        }
    };
    let scan = batches.scan;
    let mut pm = None;
    if format == Format::Parquet && !pms.is_empty() && written.is_some() {
        let pm_name = format!("{}{PM_EXT}", name.strip_suffix(TRACE_EXT).unwrap_or(name));
        let mut b = Batches {
            store: src,
            rt,
            objects: pms,
            filter: Some(("end_ns", range.0, range.1)),
            schema: Some(pm_schema()),
            columns: None,
            rank: None,
            pm_rank: sel.as_ref().and_then(|s| s.pm.as_ref()),
            scan: Scan::default(),
        };
        let opts = WriteOptions { metadata: meta, ..WriteOptions::default() };
        if let Some(w) = copy(&mut b, out, rt, &pm_name, &opts)? {
            pm = Some((w, b.scan.rows));
        }
    }
    Ok((scan, written, pm))
}

/// The batches into one Parquet object `name` (none when there are no rows).
fn copy(batches: &mut Batches<'_>, out: &Store, rt: &Handle, name: &str, opts: &WriteOptions) -> Result<Option<Written>> {
    let mut table: Option<TableWriter<ObjectWriter>> = None;
    // The processes the source files describe, for the output's footer (a later file's entry wins).
    let mut processes: BTreeMap<u32, ProcessInfo> = BTreeMap::new();
    let mut last = PathBuf::new();
    let r = batches.each(|path, meta, b| {
        if path != last {
            last = path.to_path_buf();
            let list: Vec<ProcessInfo> = meta.get(PROCESSES_META).and_then(|j| serde_json::from_str(j).ok()).unwrap_or_default();
            processes.extend(list.into_iter().map(|p| (p.pid, p)));
        }
        let t = match table.as_mut() {
            Some(t) => t,
            None => table.insert(TableWriter::new(out.writer(rt, TRACE_EXT), PathBuf::from(out.show(name)), b.schema(), opts).context(CoreSnafu)?),
        };
        t.write(&b).context(CoreSnafu)
    });
    if let Err(e) = r {
        if let Some(mut t) = table {
            t.sink_mut().abort();
        }
        return Err(e);
    }
    let Some(mut t) = table else { return Ok(None) };
    if !processes.is_empty() {
        let list: Vec<&ProcessInfo> = processes.values().collect();
        t.append_metadata(PROCESSES_META, serde_json::to_string(&list).unwrap_or_default());
    }
    let sink = t.finish().context(CoreSnafu)?;
    let bytes = sink.bytes();
    let shown = sink.finish(name)?;
    Ok(Some(Written { name: name.to_string(), shown, bytes }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use gravity_trace_core::model::{EventKind, TraceRow};
    use gravity_trace_core::parquet_io::open_reader;
    use gravity_trace_core::schema::TraceBatchBuilder;

    /// A persisted trace file of kernels starting at `starts` (row groups of 1024), named by its span like the
    /// daemon's.
    fn persisted(store: &Store, rt: &Handle, starts: std::ops::Range<i64>) {
        let mut b = TraceBatchBuilder::default();
        for t in starts.clone() {
            let ns = t * 1000;
            b.push(&TraceRow {
                kind: Some(EventKind::Kernel),
                name: "k",
                start_ns: ns,
                end_ns: ns + 10,
                pid: 1,
                ns_pid: Some(1),
                device_id: Some(0),
                stream_id: Some(7),
                correlation_id: 1,
                ..Default::default()
            });
        }
        let batch = b.finish().expect("batch");
        let wo = WriteOptions { row_group_rows: 1024, ..WriteOptions::trace() };
        let mut t = TableWriter::new(store.writer(rt, TRACE_EXT), PathBuf::from(store.location()), trace_schema(), &wo).expect("writer");
        t.write(&batch).expect("write");
        let span = (starts.start * 1000, (starts.end - 1) * 1000 + 10);
        t.finish().expect("finish").finish(&span_name(span, TRACE_EXT)).expect("publish");
    }

    fn starts(p: &Path) -> Vec<i64> {
        use arrow_array::cast::AsArray;
        use arrow_array::types::Int64Type;
        let (_, _, batches) = open_reader(p).expect("read back");
        let mut v: Vec<i64> = batches
            .flat_map(|b| b.expect("batch").column_by_name("start_ns").expect("start").as_primitive::<Int64Type>().values().to_vec())
            .collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn a_range_across_files_and_row_groups() {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(1).enable_all().build().expect("runtime");
        let h = rt.handle();
        let dir = tempfile::tempdir().expect("tempdir");
        let store = Store::open(&dir.path().display().to_string()).expect("store");
        // three persisted files, small row groups so statistics can skip some
        for chunk in [0..3000i64, 3000..6000, 6000..9000] {
            persisted(&store, h, chunk);
        }
        let files = rt.block_on(files::list(&store)).expect("list");
        assert_eq!(files.len(), 3);
        assert_eq!(files[0].span, (0, 2_999_010));

        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = Store::open(&out_dir.path().display().to_string()).expect("out");
        let range = (2_500_000, 3_500_000); // [2500, 3500) x 1000: the end of file 1, the start of file 2
        let picked: Vec<Object> = files.iter().filter(|f| f.span.0 < range.1 && f.span.1 >= range.0).map(StoredFile::object).collect();
        assert_eq!(picked.len(), 2, "file 3 is not opened");
        let (_, name) = output(Format::Parquet, &out_dir.path().display().to_string(), range);
        let (scan, written, pm) = write(Format::Parquet, &store, h, &picked, &[], range, &out, &name, None).expect("dump");
        let written = written.expect("rows");
        assert_eq!(written.shown, format!("{}/2500000-3500000.parquet", out_dir.path().display()));
        assert!(pm.is_none());
        assert_eq!(scan.rows, 1000);
        assert!(scan.row_groups_skipped >= 2, "{scan:?}");
        assert_eq!(starts(Path::new(&written.shown)), (2500..3500).map(|t| t * 1000).collect::<Vec<_>>());

        let (_, name) = output(Format::Perfetto, &out_dir.path().display().to_string(), range);
        let (scan, written, _) = write(Format::Perfetto, &store, h, &picked, &[], range, &out, &name, None).expect("perfetto");
        assert_eq!(scan.rows, 1000);
        let written = written.expect("rows");
        assert_eq!(written.shown, format!("{}/2500000-3500000.json.gz", out_dir.path().display()));
        let gz = std::fs::read(&written.shown).expect("read");
        assert_eq!(&gz[..2], &[0x1f, 0x8b], "gzip");
        let mut text = String::new();
        std::io::Read::read_to_string(&mut flate2::read::GzDecoder::new(&gz[..]), &mut text).expect("gunzip");
        assert!(text.contains("\"traceEvents\""));
        assert_eq!(text.matches("\"ph\":\"X\"").count(), 1000, "one event per activity");

        let (_, written, _) = write(Format::Parquet, &store, h, &picked, &[], (1, 2), &out, "empty.parquet", None).expect("empty");
        assert!(written.is_none(), "no rows, no file");
        let left: Vec<_> = std::fs::read_dir(out_dir.path()).expect("ls").flatten().map(|e| e.file_name()).collect();
        assert_eq!(left.len(), 2, "no temporaries: {left:?}");
    }

    #[test]
    fn rank_dump_keeps_the_devices_and_their_processes() {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(1).enable_all().build().expect("runtime");
        let h = rt.handle();
        let dir = tempfile::tempdir().expect("tempdir");
        let store = Store::open(&dir.path().display().to_string()).expect("store");
        // One persisted file: kernels of pid 1 on device 0 and pid 2 on device 1, API calls of pids 1 and 3.
        let mut b = TraceBatchBuilder::default();
        let row = |kind, pid, dev, ns| TraceRow {
            kind: Some(kind),
            name: "x",
            start_ns: ns,
            end_ns: ns + 10,
            pid,
            ns_pid: Some(pid),
            device_id: dev,
            stream_id: dev,
            correlation_id: 1,
            ..Default::default()
        };
        b.push(&row(EventKind::Kernel, 1, Some(0), 1000));
        b.push(&row(EventKind::Kernel, 2, Some(1), 2000));
        b.push(&row(EventKind::Runtime, 1, None, 3000));
        b.push(&row(EventKind::Runtime, 3, None, 4000));
        let batch = b.finish().expect("batch");
        let mut wo = WriteOptions { row_group_rows: 1024, ..WriteOptions::trace() };
        wo.metadata.insert("gpus".into(), serde_json::to_string(&[("aaa", 0u32), ("bbb", 1u32)]).expect("json"));
        let mut t = TableWriter::new(store.writer(h, TRACE_EXT), PathBuf::from(store.location()), trace_schema(), &wo).expect("writer");
        t.write(&batch).expect("write");
        t.finish().expect("finish").finish(&span_name((1000, 4010), TRACE_EXT)).expect("publish");
        // And its PM samples: "aaa" (device 0) and "bbb" (device 1).
        {
            use arrow_array::builder::StringDictionaryBuilder;
            use arrow_array::{Float64Array, Int64Array, StringArray};
            use arrow_array::types::Int32Type;
            use std::sync::Arc;
            let mut metric = StringDictionaryBuilder::<Int32Type>::new();
            for _ in 0..3 {
                metric.append_value("m");
            }
            let samples = arrow_array::RecordBatch::try_new(
                pm_schema(),
                vec![
                    Arc::new(StringArray::from(vec!["aaa", "bbb", "ccc"])),
                    Arc::new(Int64Array::from(vec![1500, 2500, 3500])),
                    Arc::new(metric.finish()),
                    Arc::new(Float64Array::from(vec![0.1, 0.2, 0.3])),
                ],
            )
            .expect("pm batch");
            let mut t = TableWriter::new(store.writer(h, PM_EXT), PathBuf::from(store.location()), pm_schema(), &WriteOptions::default()).expect("pm writer");
            t.write(&samples).expect("write");
            t.finish().expect("finish").finish(&span_name((1500, 3500), PM_EXT)).expect("publish");
        }

        let files = rt.block_on(files::list(&store)).expect("list");
        let picked: Vec<Object> = files.iter().filter(|f| f.kind == FileKind::Trace).map(StoredFile::object).collect();
        let pms: Vec<Object> = files.iter().filter(|f| f.kind == FileKind::Pm).map(StoredFile::object).collect();
        assert_eq!((picked.len(), pms.len()), (1, 1));
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = Store::open(&out_dir.path().display().to_string()).expect("out");
        let ranks = Ranks::parse("0").expect("rank");
        let (scan, written, pm) =
            write(Format::Parquet, &store, h, &picked, &pms, (0, 5000), &out, "rank0.parquet", Some(&ranks)).expect("dump");
        let written = written.expect("rows");
        assert_eq!(scan.rows, 2, "the device-0 kernel and pid 1's API call, nothing else");
        let rows = trace_rows(Path::new(&written.shown));
        assert_eq!(rows, vec![(1, None, 3000), (1, Some(0), 1000)], "sorted by pid, device (nulls first), start");

        // The PM samples: only "aaa" (device 0), renamed to "rank 0"; "ccc" has no known device.
        let (pm_written, pm_rows) = pm.expect("pm file");
        assert_eq!(pm_rows, 1);
        let (_, _, batches) = open_reader(Path::new(&pm_written.shown)).expect("read pm");
        let names: Vec<String> = batches
            .flat_map(|b| {
                let b = b.expect("batch");
                let u = b.column_by_name("gpu_uuid").expect("uuid").as_any().downcast_ref::<arrow_array::StringArray>().expect("string").clone();
                (0..b.num_rows()).map(move |i| u.value(i).to_string()).collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(names, ["rank 0"], "the device's PM samples, named by its rank");
    }

    /// (pid, device_id, start_ns) of a trace file's rows.
    fn trace_rows(p: &Path) -> Vec<(u32, Option<u32>, i64)> {
        let (_, _, batches) = open_reader(p).expect("read back");
        batches
            .flat_map(|b| {
                let b = b.expect("batch");
                use arrow_array::Array;
                use arrow_array::cast::AsArray;
                use arrow_array::types::{Int64Type, UInt32Type};
                let pid = b.column_by_name("pid").expect("pid").as_primitive::<UInt32Type>().clone();
                let dev = b.column_by_name("device_id").expect("device").as_primitive::<UInt32Type>().clone();
                let start = b.column_by_name("start_ns").expect("start").as_primitive::<Int64Type>().clone();
                (0..b.num_rows()).map(move |i| (pid.value(i), if dev.is_valid(i) { Some(dev.value(i)) } else { None }, start.value(i)))
            })
            .collect()
    }

    #[test]
    fn output_names() {
        let r = (1, 2);
        let o = |f, s: &str| output(f, s, r);
        assert_eq!(o(Format::Parquet, "/o"), ("/o".into(), "1-2.parquet".into()));
        assert_eq!(o(Format::Parquet, "/o/x.parquet"), ("/o".into(), "x.parquet".into()));
        assert_eq!(o(Format::Parquet, "x.parquet"), (".".into(), "x.parquet".into()));
        assert_eq!(o(Format::Perfetto, "/o"), ("/o".into(), "1-2.json.gz".into()));
        assert_eq!(o(Format::Perfetto, "/o/x.json"), ("/o".into(), "x.json.gz".into()), "always gzip");
        assert_eq!(o(Format::Perfetto, "s3://b/p/x.json.gz"), ("s3://b/p".into(), "x.json.gz".into()));
        assert_eq!(local_addr("0.0.0.0:9465"), "127.0.0.1:9465");
        assert_eq!(local_addr(":9465"), "127.0.0.1:9465");
        assert_eq!(local_addr("10.0.0.1:9465"), "10.0.0.1:9465");
    }
}
