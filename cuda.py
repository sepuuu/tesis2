import onnxruntime as ort
sess = ort.InferenceSession("codes/models/players.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("Proveedor activo:", sess.get_providers())


import onnxruntime as ort
print(ort.__version__)
print("Proveedores disponibles:", ort.get_available_providers())
