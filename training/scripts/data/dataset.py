"""Dataset container with split-aware operations.

See DATA_LOADING_DOCS.md for architecture and ordering requirements.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SampleMetadata:
    """Metadata for filtering and transforms. Stripped during formatting."""
    intervention_type: str
    tags: List[str]
    original_A: Optional[str] = None
    original_B: Optional[str] = None


@dataclass
class Sample:
    """Sample data. metadata=None indicates formatted sample."""
    system_prompt: str
    task: str
    response: str
    intervention: str
    review: str
    flag: str
    variant_key: str
    metadata: Optional[SampleMetadata] = None


class Dataset:
    """Container for Sample objects. Tracks split state to enforce correct operation order."""
    def __init__(self, samples: List[Sample], is_split: bool = False):
        self.samples = list(samples)
        self._is_split = is_split

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    def filter(self, intervention_types: Optional[List[str]] = None,
               tags_to_filter: Optional[List[str]] = None) -> 'Dataset':
        """Apply filters BEFORE split (split-critical operation)."""
        if self._is_split:
            raise RuntimeError(
                "Cannot filter after split! Must filter before stratified_split() "
                "to maintain split-critical behavior."
            )

        filtered = []

        for sample in self.samples:
            if sample.metadata is None:
                filtered.append(sample)
                continue

            if intervention_types is not None:
                if sample.metadata.intervention_type not in intervention_types:
                    continue

            if tags_to_filter is not None:
                if set(tags_to_filter).intersection(sample.metadata.tags):
                    continue

            filtered.append(sample)

        return Dataset(filtered, is_split=False)

    def shuffle(self, seed: int) -> 'Dataset':
        """Shuffle dataset. Can ONLY be called after split."""
        if not self._is_split:
            raise RuntimeError(
                "Cannot shuffle before split! Call stratified_split() first."
            )

        import random
        shuffled = self.samples.copy()
        random.seed(seed)
        random.shuffle(shuffled)
        return Dataset(shuffled, is_split=True)

    def stratified_split(self, n_val: int, split: str) -> 'Dataset':
        """Apply stratified split: every Nth sample to validation."""
        if self._is_split:
            raise RuntimeError("Cannot split twice!")

        total = len(self.samples)
        if total == 0:
            return Dataset([], is_split=True)

        n_val = min(n_val, total)
        if n_val == 0:
            if split == "train":
                return Dataset(self.samples, is_split=True)
            else:
                return Dataset([], is_split=True)

        stride = total / n_val

        val_indices = set()
        for i in range(n_val):
            idx = int(i * stride)
            val_indices.add(idx)

        # Return requested split
        if split == "val":
            samples = [self.samples[i] for i in sorted(val_indices)]
        elif split == "train":
            samples = [self.samples[i] for i in range(total) if i not in val_indices]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")

        return Dataset(samples, is_split=True)

    def sample(self, n: int, seed: int) -> 'Dataset':
        """Randomly sample n items from this dataset."""
        import random
        rng = random.Random(seed)
        sampled = rng.sample(self.samples, min(n, len(self.samples)))
        return Dataset(sampled, is_split=self._is_split)

    def apply_experiment_transforms(self, mismatch_prompts: bool = False,
                                   exclude_system_prompt: bool = False) -> 'Dataset':
        """Apply experiment transforms: swap prompts or exclude system prompt."""
        if not mismatch_prompts and not exclude_system_prompt:
            return self

        transformed = []
        for sample in self.samples:
            if sample.metadata is None:
                transformed.append(sample)
                continue

            new_system_prompt = sample.system_prompt

            if mismatch_prompts:
                if sample.metadata.original_A and sample.metadata.original_B:
                    if sample.variant_key == "A":
                        new_system_prompt = sample.metadata.original_B
                    else:
                        new_system_prompt = sample.metadata.original_A

            if exclude_system_prompt:
                new_system_prompt = ""

            transformed_sample = Sample(
                system_prompt=new_system_prompt,
                task=sample.task,
                response=sample.response,
                intervention=sample.intervention,
                review=sample.review,
                flag=sample.flag,
                variant_key=sample.variant_key,
                metadata=sample.metadata
            )

            transformed.append(transformed_sample)

        return Dataset(transformed, is_split=self._is_split)

    def apply_formatting(self, system_tag: str = "<SYSTEM>",
                        intervention_prefix: str = "\n<INTERVENTION>: ",
                        review_prefix: str = "\nREVIEW:",
                        flag_prefix: str = "\nFLAG: ") -> 'Dataset':
        """Apply formatting: add prefixes and tags. Strips metadata (final step)."""
        formatted = []

        for sample in self.samples:
            if sample.metadata is None:
                formatted.append(sample)
                continue

            system_prompt_formatted = (
                f"{system_tag}{sample.system_prompt}{system_tag}"
                if sample.system_prompt else ""
            )

            formatted_sample = Sample(
                system_prompt=system_prompt_formatted,
                task=sample.task,
                response=sample.response,
                intervention=f"{intervention_prefix}{sample.intervention}",
                review=f"{review_prefix}{sample.review}",
                flag=f"{flag_prefix}{sample.flag}",
                variant_key=sample.variant_key,
                metadata=None
            )

            formatted.append(formatted_sample)

        return Dataset(formatted, is_split=self._is_split)
