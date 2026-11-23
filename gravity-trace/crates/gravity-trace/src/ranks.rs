//! The rank selection of `--rank` / `[collect] ranks`: only these CUDA devices (the SDK's `cudevice_ordinal`,
//! the trace's `device_id` column) and the processes running on them reach the trace.
//!
//! A process belongs to a rank once it produced GPU activity on one of the selected devices: API calls carry
//! no device, so they are kept by process. PM samples are kept per GPU whose device is known (learned from
//! the GPU's activity rows, so the first snapshot's samples of a GPU not yet seen active are dropped, and a
//! never-active GPU never attributes any).
use crate::error::{BadRankSnafu, Result};
use snafu::OptionExt;
use std::fmt;

/// CUDA device ordinals (sorted, unique).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ranks(Vec<u32>);

impl Ranks {
    /// `0,2,3`: comma-separated CUDA device numbers.
    pub fn parse(s: &str) -> Result<Ranks> {
        let mut devs = Vec::new();
        for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let d = part.parse::<u32>().ok().context(BadRankSnafu { input: s.to_string() })?;
            devs.push(d);
        }
        devs.sort_unstable();
        devs.dedup();
        if devs.is_empty() {
            return BadRankSnafu { input: s.to_string() }.fail();
        }
        Ok(Ranks(devs))
    }

    /// `[collect] ranks` (`["0,2", "3"]` and `["0", "2", "3"]` alike); an empty list = no filter.
    pub fn from_config(list: &[String]) -> Result<Option<Ranks>> {
        match list.is_empty() {
            true => Ok(None),
            false => Self::parse(&list.join(",")).map(Some),
        }
    }

    /// The devices, ascending.
    pub fn devices(&self) -> &[u32] {
        &self.0
    }
}

impl fmt::Display for Ranks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = self.0.iter();
        if let Some(d) = parts.next() {
            write!(f, "{d}")?;
        }
        for d in parts {
            write!(f, ",{d}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_lists() {
        assert_eq!(Ranks::parse("0,2, 3").expect("valid").devices(), &[0, 2, 3]);
        assert_eq!(Ranks::parse("2,0,2").expect("valid").devices(), &[0, 2], "sorted, deduplicated");
        assert_eq!(Ranks::parse("7").expect("valid").to_string(), "7");
        assert!(Ranks::parse("").is_err() && Ranks::parse(",").is_err() && Ranks::parse("x").is_err());
        assert!(Ranks::parse("1-3").is_err(), "ranges are not a thing: 1,2,3");
        assert_eq!(Ranks::from_config(&[]).expect("none"), None);
        let Some(r) = Ranks::from_config(&["0,1".into()]).expect("some") else { panic!("a list becomes a filter") };
        assert_eq!(r.devices(), &[0, 1]);
    }
}
