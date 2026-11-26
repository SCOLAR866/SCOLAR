import sys
import importlib.util
import pathlib
from colorama import Fore

# paths
# the target is where transformers thinks the file lives
target_module_name = 'transformers.models.qwen2_5_vl.modeling_qwen2_5_vl'
# patch file
patch_path = pathlib.Path(__file__).parent / 'my_modeling_qwen2_5_vl.py'

patches = [
    {
        # the target is where transformers thinks the file lives
        'target_module_name': 'transformers.models.qwen2_5_vl.configuration_qwen2_5_vl',
        # patch file
        'patch_path': pathlib.Path(__file__).parent / 'my_configuration_qwen2_5_vl.py',
    }, # CRITICAL: the configuration file should be patched earlier
    {
        # the target is where transformers thinks the file lives
        'target_module_name': 'transformers.models.qwen2_5_vl.modeling_qwen2_5_vl',
        # patch file
        'patch_path': pathlib.Path(__file__).parent / 'my_modeling_qwen2_5_vl.py',
    },
]

def apply_qwen2_5_vl_patch(target_module_name, patch_path):
    # create the spec and module
    spec = importlib.util.spec_from_file_location(target_module_name, str(patch_path))
    patched_mod = importlib.util.module_from_spec(spec)

    # inject into sys.modules before executing
    sys.modules[target_module_name] = patched_mod

    try:
        # execute the module code
        spec.loader.exec_module(patched_mod)
        print(f'{Fore.BLUE}Successfully replaced the Qwen2_5_VL module {target_module_name}.{Fore.RESET}')
    except Exception as e:
        if target_module_name in sys.modules:
            del sys.modules[target_module_name]
        print(f'{Fore.RED}Failed to patch {target_module_name}. {e} {Fore.RESET}')
        raise e

for patch in patches:
    apply_qwen2_5_vl_patch(patch['target_module_name'], patch['patch_path'])
