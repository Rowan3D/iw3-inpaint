r"""
Extras for iw3: a trained inpainting model, the mask it needs, and full-size
output from a low-resolution inpainting pass.

Installed by install.bat into <nunif>\nt_inpaint\ and switched on by a two-line
hook in site-packages. Nothing inside nunif\ or iw3\ is modified, so a nunif
update cannot conflict with it and deleting the folder removes it completely.
"""
__all__ = ["boot", "iw3_mask", "lowres", "models"]
__version__ = "1.0"
