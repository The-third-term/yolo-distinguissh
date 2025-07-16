import acl

class Model:
    def __init__(self, model_path, device_id=0):
        self.device_id = device_id
        self.model_path = model_path
        self.model_id = None
        self.desc = None
        self.context = None

    def init(self):
        acl.init()
        self.context, self.stream = acl.rt.create_context(self.device_id)
        self.model_id = acl.mdl.load_from_file(self.model_path)
        self.desc = acl.mdl.create_desc()
        acl.mdl.get_desc(self.desc, self.model_id)

    def infer(self, input_array):
        # 这里的输入数据应是 numpy array，需和模型输入一致
        # 这里只演示流程，具体按实际格式 reshape
        import numpy as np

        input_bytes = input_array.nbytes
        input_buffer, input_dev = acl.rt.malloc_host_and_device(input_bytes)
        np.copyto(np.frombuffer(input_buffer, dtype=np.float32).reshape(input_array.shape), input_array)
        acl.rt.memcpy(input_dev, input_bytes, input_buffer, input_bytes, acl.ACL_MEMCPY_HOST_TO_DEVICE)

        input_dataset = acl.mdl.create_dataset([input_dev])
        output_dataset = acl.mdl.create_dataset_with_desc(self.desc)

        acl.mdl.execute(self.model_id, input_dataset, output_dataset)

        output_buffer = acl.mdl.get_dataset_buffer(output_dataset, 0)
        output_host, _ = acl.rt.malloc_host(acl.mdl.get_dataset_size(output_dataset))
        acl.rt.memcpy(output_host, acl.mdl.get_dataset_size(output_dataset), output_buffer, acl.mdl.get_dataset_size(output_dataset), acl.ACL_MEMCPY_DEVICE_TO_HOST)

        output_data = np.frombuffer(output_host, dtype=np.float32)
        return output_data

    def release(self):
        acl.rt.destroy_context(self.context)
        acl.finalize()
