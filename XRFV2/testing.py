
from init_utils import init_dataset, init_model, init_test_dataset
from model.models import make_model, make_model_config
from pipeline.tester import Tester

def test(config):
    test_dataset = init_test_dataset(config)


    model_cfg = make_model_config(config['model']['backbone_name'], config['model'])
    model = make_model(config['model']['name'], model_cfg, label_desc_type=config['label_desc_type'], model_key=config['embeding_mode_name'])

    tester = Tester(config, test_dataset=test_dataset, model=model)
    tester.testing()