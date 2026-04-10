from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HistoricalWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_year: int
    end_year: int
    label: str
    major_events: list[str] = Field(default_factory=list)
    available_tech: list[str] = Field(default_factory=list)
    unavailable_tech: list[str] = Field(default_factory=list)
    cultural_norms: list[str] = Field(default_factory=list)
    media_landscape: list[str] = Field(default_factory=list)


class TechFamiliarity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grew_up_with: list[str] = Field(default_factory=list)
    adopted_as_adult: list[str] = Field(default_factory=list)
    never_used: list[str] = Field(default_factory=list)
    comfort_level: dict[str, str] = Field(default_factory=dict)


class CulturalReferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    music: list[str] = Field(default_factory=list)
    tv_and_film: list[str] = Field(default_factory=list)
    slang_era: str
    brands: list[str] = Field(default_factory=list)
    news_events: list[str] = Field(default_factory=list)
    shared_experiences: list[str] = Field(default_factory=list)


class EconomicAssumptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_market_at_entry: str
    housing_expectation: str
    retirement_model: str
    debt_norms: str
    career_model: str


class SlangProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_slang: list[str] = Field(default_factory=list)
    recognized_slang: list[str] = Field(default_factory=list)
    unknown_slang: list[str] = Field(default_factory=list)


class CohortModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    birth_year: int = Field(ge=1920, le=2100)
    eval_year: int = Field(ge=1920, le=2100)
    childhood_window: HistoricalWindow
    adolescence_window: HistoricalWindow
    early_adulthood_window: HistoricalWindow
    major_events_lived_through: list[str] = Field(default_factory=list)
    tech_familiarity: TechFamiliarity
    cultural_references: CulturalReferences
    economic_assumptions: EconomicAssumptions
    slang_compatibility: SlangProfile

    @property
    def cohort_label(self) -> str:
        return f"{self.birth_year}s_birth_cohort"


class DecadeReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decade: str
    start_year: int | None = None
    end_year: int | None = None
    major_events: list[str] = Field(default_factory=list)
    technology: list[str] = Field(default_factory=list)
    cultural_touchstones: list[str] = Field(default_factory=list)
    economic_conditions: str
    family_norms: str
    communication_tech: list[str] = Field(default_factory=list)
    dominant_media: list[str] = Field(default_factory=list)
    slang: list[str] = Field(default_factory=list)
    work_norms: str
    brands: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fill_year_bounds(self) -> "DecadeReference":
        if self.start_year is not None and self.end_year is not None:
            return self
        decade_base = int(self.decade[:4])
        self.start_year = decade_base
        self.end_year = decade_base + 9
        return self


@lru_cache(maxsize=1)
def load_decade_references() -> list[DecadeReference]:
    data_path = Path(__file__).resolve().parents[1] / "data" / "historical_decades.json"
    raw = json.loads(data_path.read_text())
    return [DecadeReference.model_validate(item) for item in raw]


def load_default_decades() -> list[DecadeReference]:
    return load_decade_references()


class CohortBuilder:
    def __init__(self, decades_data: list[DecadeReference] | None = None) -> None:
        self.decades = decades_data or load_default_decades()
        self.decades_data = self.decades

    def build(self, birth_year: int, eval_year: int) -> CohortModel:
        childhood = self._build_window(birth_year, birth_year + 12, "childhood")
        adolescence = self._build_window(birth_year + 13, birth_year + 17, "adolescence")
        early_adulthood = self._build_window(birth_year + 18, birth_year + 25, "early_adulthood")
        all_lived = self._collect_events(birth_year, eval_year)

        native_tech = self._collect_technology(childhood.start_year, adolescence.end_year)
        adult_tech = [
            tech
            for tech in self._collect_technology(early_adulthood.start_year, eval_year)
            if tech not in native_tech
        ]
        future_tech = [
            tech
            for ref in self.decades_data
            if ref.start_year > eval_year
            for tech in ref.technology
        ]

        comfort_level = {tech: "native" for tech in native_tech}
        for tech in adult_tech:
            comfort_level[tech] = "competent"
        for tech in future_tech:
            comfort_level[tech] = "unaware"

        culture_refs = CulturalReferences(
            music=adolescence.media_landscape[:4],
            tv_and_film=early_adulthood.media_landscape[:4],
            slang_era=", ".join(adolescence.media_landscape[:2]) or adolescence.label,
            brands=list(
                dict.fromkeys(
                    brand
                    for ref in self._references_for_range(
                        early_adulthood.start_year,
                        early_adulthood.end_year,
                    )
                    for brand in ref.brands
                )
            )[:4],
            news_events=all_lived[:6],
            shared_experiences=(adolescence.major_events + early_adulthood.major_events)[:6],
        )

        entry_ref = self._dominant_decade(early_adulthood.start_year, early_adulthood.end_year)
        economic = EconomicAssumptions(
            job_market_at_entry=entry_ref.economic_conditions,
            housing_expectation=self._housing_expectation(entry_ref),
            retirement_model=self._retirement_model(entry_ref),
            debt_norms=self._debt_norms(entry_ref),
            career_model=self._career_model(entry_ref),
        )

        recognized = list(dict.fromkeys(adolescence.media_landscape[:2] + early_adulthood.media_landscape[:2]))
        future_slang = [
            slang
            for ref in self.decades_data
            if ref.start_year > eval_year
            for slang in ref.slang
        ]
        slang = SlangProfile(
            active_slang=adolescence.media_landscape[:0] + adolescence.major_events[:0] or adolescence.media_landscape[:0],
            recognized_slang=recognized,
            unknown_slang=future_slang[:6],
        )
        if not slang.active_slang:
            slang.active_slang = entry_ref.slang[:4]
        if not slang.recognized_slang:
            slang.recognized_slang = entry_ref.slang[:4]

        return CohortModel(
            birth_year=birth_year,
            eval_year=eval_year,
            childhood_window=childhood,
            adolescence_window=adolescence,
            early_adulthood_window=early_adulthood,
            major_events_lived_through=all_lived,
            tech_familiarity=TechFamiliarity(
                grew_up_with=native_tech[:12],
                adopted_as_adult=adult_tech[:12],
                never_used=future_tech[:12],
                comfort_level=comfort_level,
            ),
            cultural_references=culture_refs,
            economic_assumptions=economic,
            slang_compatibility=SlangProfile(
                active_slang=entry_ref.slang[:4],
                recognized_slang=recognized[:6] or entry_ref.slang[:4],
                unknown_slang=future_slang[:6],
            ),
        )

    @staticmethod
    def build_label(birth_year: int) -> str:
        if 1981 <= birth_year <= 1996:
            return f"millennial ({birth_year}-born)"
        if 1965 <= birth_year <= 1980:
            return f"gen_x ({birth_year}-born)"
        if birth_year <= 1964:
            return f"boomer_or_older ({birth_year}-born)"
        if 1997 <= birth_year <= 2012:
            return f"gen_z ({birth_year}-born)"
        return f"gen_alpha_or_later ({birth_year}-born)"

    def _build_window(self, start_year: int, end_year: int, label: str) -> HistoricalWindow:
        refs = self._references_for_range(start_year, end_year)
        available_tech = list(dict.fromkeys(tech for ref in refs for tech in ref.technology))
        future_refs = [ref for ref in self.decades_data if ref.start_year > end_year]
        unavailable_tech = list(dict.fromkeys(tech for ref in future_refs[:2] for tech in ref.technology))
        return HistoricalWindow(
            start_year=start_year,
            end_year=end_year,
            label=label,
            major_events=list(dict.fromkeys(event for ref in refs for event in ref.major_events))[:8],
            available_tech=available_tech[:8],
            unavailable_tech=unavailable_tech[:8],
            cultural_norms=list(dict.fromkeys(ref.family_norms for ref in refs))[:4],
            media_landscape=list(dict.fromkeys(item for ref in refs for item in ref.cultural_touchstones))[:8],
        )

    def _collect_events(self, start_year: int, end_year: int) -> list[str]:
        return list(dict.fromkeys(event for ref in self._references_for_range(start_year, end_year) for event in ref.major_events))

    def _collect_technology(self, start_year: int, end_year: int) -> list[str]:
        return list(dict.fromkeys(tech for ref in self._references_for_range(start_year, end_year) for tech in ref.technology))

    def _references_for_range(self, start_year: int, end_year: int) -> list[DecadeReference]:
        return [
            ref
            for ref in self.decades_data
            if not (ref.end_year < start_year or ref.start_year > end_year)
        ]

    def _dominant_decade(self, start_year: int, end_year: int) -> DecadeReference:
        refs = self._references_for_range(start_year, end_year)
        return refs[0]

    @staticmethod
    def _housing_expectation(ref: DecadeReference) -> str:
        text = ref.economic_conditions.lower()
        if "boom" in text or "growth" in text:
            return "stretch but reachable"
        if "depression" in text or "recession" in text or "uncertain" in text:
            return "precarious"
        return "affordable with sacrifice"

    @staticmethod
    def _retirement_model(ref: DecadeReference) -> str:
        text = ref.work_norms.lower()
        if "loyalty" in text or "company" in text:
            return "pension_or_long_tenure"
        if "portfolio" in text or "gig" in text:
            return "self_funded_and_uncertain"
        return "401k_or_equivalent"

    @staticmethod
    def _debt_norms(ref: DecadeReference) -> str:
        text = ref.economic_conditions.lower()
        if "depression" in text:
            return "debt_avoidance"
        if "credit" in text or "consumer" in text:
            return "consumer_debt_normalized"
        return "mortgage_first"

    @staticmethod
    def _career_model(ref: DecadeReference) -> str:
        text = ref.work_norms.lower()
        if "company loyalty" in text or "hierarchy" in text or "industrial" in text:
            return "company_man"
        if "promotion" in text or "ladder" in text:
            return "ladder_climb"
        if "portfolio" in text or "gig" in text or "creator" in text:
            return "portfolio_career"
        return "job_hopping"


__all__ = [
    "CohortBuilder",
    "CohortModel",
    "CulturalReferences",
    "DecadeReference",
    "EconomicAssumptions",
    "HistoricalWindow",
    "SlangProfile",
    "TechFamiliarity",
    "load_decade_references",
    "load_default_decades",
]
