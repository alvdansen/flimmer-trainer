"""Auto-register all model definitions on import.

Each model module constructs a ModelDefinition and calls register_model()
at module level. Importing this package triggers all registrations.
"""

from . import moe  # noqa: F401
from . import wan21_i2v  # noqa: F401
from . import wan21_t2v  # noqa: F401
from . import wan22_i2v  # noqa: F401
