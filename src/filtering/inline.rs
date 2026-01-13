//! Inline filtering strategies for vector search.

use super::{FilterPredicate, MetadataStore};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterStrategy {
    PreFilter,
    PostFilter,
    Inline,
}

#[derive(Clone, Debug)]
pub struct InlineFilterConfig {
    pub post_filter_oversearch: f32,
    pub inline_max_candidates: usize,
    pub pre_filter_threshold: f32,
    pub post_filter_threshold: f32,
}

impl Default for InlineFilterConfig {
    fn default() -> Self {
        Self {
            post_filter_oversearch: 4.0,
            inline_max_candidates: 10_000,
            pre_filter_threshold: 0.05,
            post_filter_threshold: 0.5,
        }
    }
}

pub struct FilterStrategySelector {
    config: InlineFilterConfig,
}

impl FilterStrategySelector {
    pub fn new() -> Self {
        Self {
            config: InlineFilterConfig::default(),
        }
    }

    pub fn with_config(config: InlineFilterConfig) -> Self {
        Self { config }
    }

    pub fn select(
        &self,
        estimated_selectivity: f32,
        k: usize,
        total_docs: usize,
    ) -> FilterStrategy {
        if estimated_selectivity < self.config.pre_filter_threshold {
            return FilterStrategy::PreFilter;
        }
        if estimated_selectivity > self.config.post_filter_threshold {
            return FilterStrategy::PostFilter;
        }

        let expected_matches = (total_docs as f32 * estimated_selectivity) as usize;
        if k * 10 < expected_matches {
            return FilterStrategy::PostFilter;
        }

        FilterStrategy::Inline
    }

    pub fn estimate_selectivity(
        &self,
        filter: &FilterPredicate,
        metadata: &MetadataStore,
        total_docs: usize,
    ) -> f32 {
        if total_docs == 0 {
            return 1.0;
        }
        estimate_predicate_selectivity(filter, metadata, total_docs)
    }
}

impl Default for FilterStrategySelector {
    fn default() -> Self {
        Self::new()
    }
}

fn estimate_predicate_selectivity(
    predicate: &FilterPredicate,
    metadata: &MetadataStore,
    total_docs: usize,
) -> f32 {
    match predicate {
        FilterPredicate::Equals { field, value } => {
            let counts = metadata.get_value_counts(field);
            let matching = counts
                .iter()
                .find(|(v, _)| *v == *value)
                .map(|(_, count)| *count)
                .unwrap_or(0);
            matching as f32 / total_docs as f32
        }
        FilterPredicate::And(predicates) => predicates.iter().fold(1.0, |acc, p| {
            acc * estimate_predicate_selectivity(p, metadata, total_docs)
        }),
        FilterPredicate::Or(predicates) => {
            if predicates.is_empty() {
                return 0.0;
            }
            let non_selectivity = predicates.iter().fold(1.0, |acc, p| {
                acc * (1.0 - estimate_predicate_selectivity(p, metadata, total_docs))
            });
            1.0 - non_selectivity
        }
    }
}

pub struct InlineFilter<'a> {
    predicate: &'a FilterPredicate,
    metadata: &'a MetadataStore,
    evaluated: usize,
    passed: usize,
}

impl<'a> InlineFilter<'a> {
    pub fn new(predicate: &'a FilterPredicate, metadata: &'a MetadataStore) -> Self {
        Self {
            predicate,
            metadata,
            evaluated: 0,
            passed: 0,
        }
    }

    pub fn matches(&mut self, doc_id: u32) -> bool {
        self.evaluated += 1;
        let result = self.metadata.matches(doc_id, self.predicate);
        if result {
            self.passed += 1;
        }
        result
    }

    pub fn observed_selectivity(&self) -> f32 {
        if self.evaluated == 0 {
            1.0
        } else {
            self.passed as f32 / self.evaluated as f32
        }
    }

    pub fn stats(&self) -> InlineFilterStats {
        InlineFilterStats {
            evaluated: self.evaluated,
            passed: self.passed,
            selectivity: self.observed_selectivity(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct InlineFilterStats {
    pub evaluated: usize,
    pub passed: usize,
    pub selectivity: f32,
}

pub fn post_filter_results(
    results: Vec<(u32, f32)>,
    predicate: &FilterPredicate,
    metadata: &MetadataStore,
    k: usize,
) -> Vec<(u32, f32)> {
    results
        .into_iter()
        .filter(|(id, _)| metadata.matches(*id, predicate))
        .take(k)
        .collect()
}

pub fn calculate_oversearch_k(
    k: usize,
    estimated_selectivity: f32,
    oversearch_factor: f32,
) -> usize {
    if estimated_selectivity <= 0.0 {
        return k * 100;
    }
    let base_oversearch = (k as f32 / estimated_selectivity).ceil() as usize;
    let with_factor = (base_oversearch as f32 * oversearch_factor).ceil() as usize;
    with_factor.max(k).min(k * 100)
}
