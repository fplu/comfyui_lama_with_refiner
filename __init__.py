import folder_paths
import os


def _add_folder_path(folder_name: str, extensions_to_register: list):
    path = os.path.join(folder_paths.models_dir, folder_name)
    folders, extensions = folder_paths.folder_names_and_paths.get(folder_name, ([], set()))
    if path not in folders:
        folders.append(path)
    if isinstance(extensions, set):
        extensions.update(extensions_to_register)
    elif isinstance(extensions, list):
        extensions.extend(extensions_to_register)
    else:
        e = f"Failed to register models/inpaint folder. Found existing value: {extensions}"
        raise Exception(e)
    folder_paths.folder_names_and_paths[folder_name] = (folders, extensions)


_add_folder_path("inpaint", [".pt", ".pth", ".safetensors", ".patch"])

from . import nodes

NODE_CLASS_MAPPINGS = {
    "INPAINT_LoadInpaintLaMaModel": nodes.LoadInpaintLaMaModel,
    "INPAINT_InpaintWithLaMaRefinerModel": nodes.InpaintWithLamaRefinerModel,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "INPAINT_LoadInpaintLaMaModel": "Load Inpaint LaMa Model",
    "INPAINT_InpaintWithLaMaRefinerModel": "Inpaint (using LaMa + Refinement)",
}
