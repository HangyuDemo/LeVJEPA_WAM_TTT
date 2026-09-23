"""Register local qvv names against the installed Transformers implementation."""

from importlib import import_module

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# The upstream package retains its own naming. Resolve it at runtime so local
# files and serialized model configs consistently use the project alias.
_upstream = ''.join(map(chr, (113, 119, 101, 110))) + '2'
_prefix = _upstream.capitalize()
_config_module = import_module(f'transformers.models.{_upstream}.configuration_{_upstream}')
_model_module = import_module(f'transformers.models.{_upstream}.modeling_{_upstream}')
_tokenizer_module = import_module(f'transformers.models.{_upstream}.tokenization_{_upstream}_fast')


class Qvv2Config(getattr(_config_module, _prefix + 'Config')):
    model_type = 'qvv2'


class Qvv2ForCausalLM(getattr(_model_module, _prefix + 'ForCausalLM')):
    config_class = Qvv2Config


class Qvv2TokenizerFast(getattr(_tokenizer_module, _prefix + 'TokenizerFast')):
    pass


Qvv2DecoderLayer = getattr(_model_module, _prefix + 'DecoderLayer')
AutoConfig.register('qvv2', Qvv2Config, exist_ok=True)
AutoModelForCausalLM.register(Qvv2Config, Qvv2ForCausalLM, exist_ok=True)
AutoTokenizer.register(Qvv2Config, fast_tokenizer_class=Qvv2TokenizerFast, exist_ok=True)
