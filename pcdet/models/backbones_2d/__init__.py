from pcdet.pointpillar_quantize_config import load_rpn_config_json
config_dict = load_rpn_config_json.get_config()
quantize_mode = config_dict["quantize_mode"]

<<<<<<< HEAD
if quantize_mode == "true":
=======
if quantize_mode == "True":
>>>>>>> 7a56810aef3b30c025d8a9281f6ad7646b288683
    from .base_bev_backbone_for_calib import BaseBEVBackbone
else:
    from .base_bev_backbone import BaseBEVBackbone
__all__ = {
    'BaseBEVBackbone': BaseBEVBackbone
}
