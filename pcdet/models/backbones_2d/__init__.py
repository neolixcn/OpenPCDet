from pcdet.pointpillar_quantize_config import load_rpn_config_json
config_dict = load_rpn_config_json.get_config()
quantize_mode = config_dict["quantize_mode"]

if quantize_mode == "true":
    from .base_bev_backbone_for_calib import BaseBEVBackbone
else:
    from .base_bev_backbone import BaseBEVBackbone
__all__ = {
    'BaseBEVBackbone': BaseBEVBackbone
}
