"""
This module contains 'launchers', which are self-contained functions that take
one dictionary and run a full experiment. The dictionary configures the
experiment.

It is important that the functions are completely self-contained (i.e. they
import their own modules) so that they can be serialized.
"""
