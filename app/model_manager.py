import os
from pathlib import Path

class ModelManager:
    @staticmethod
    def find_models(models_folder):
        """
        Возвращает список путей к папкам, которые можно считать моделями OpenVINO.
        Папка считается моделью, если в ней есть хотя бы один .xml и один .bin файл,
        или файл с именем openvino_model.xml, или файл с именем model.xml.
        """
        models = []
        if not models_folder or not os.path.isdir(models_folder):
            return models

        for root, dirs, files in os.walk(models_folder):
            has_xml = any(f.endswith('.xml') for f in files)
            has_bin = any(f.endswith('.bin') for f in files)
            has_openvino_model = 'openvino_model.xml' in files
            has_model_xml = 'model.xml' in files

            if (has_xml and has_bin) or has_openvino_model or has_model_xml:
                models.append(root)

        def model_priority(path):
            name = os.path.basename(path).lower()
            if "turbo-int8" in name:
                return 0, name
            if "turbo-fp16" in name:
                return 1, name
            return 2, name

        return sorted(models, key=model_priority)

    @staticmethod
    def get_model_name(model_path):
        return os.path.basename(model_path.rstrip(os.sep))
