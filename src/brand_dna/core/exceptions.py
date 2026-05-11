"""Domain-specific exceptions. Caught and logged at orchestrator boundary."""


class BrandDNAError(Exception):
    """Base exception. All recoverable failures derive from this."""


class DiscoveryError(BrandDNAError):
    """Failure to discover content on a brand's web presence."""


class AcquisitionError(BrandDNAError):
    """Failure to acquire images or text from a discovered source."""


class FilteringError(BrandDNAError):
    """Failure in quality / fashion / dedup filtering."""


class AnalysisError(BrandDNAError):
    """Failure in vision or text analysis."""


class SynthesisError(BrandDNAError):
    """Failure during dossier synthesis or PDF rendering."""


class LLMError(BrandDNAError):
    """Failure communicating with an LLM provider."""


class ConfigurationError(BrandDNAError):
    """Invalid or missing configuration."""
