import coremltools

from coremltools.models.neural_network.quantization_utils import *
mode_name = "food"

model = coremltools.models.MLModel(mode_name+".mlmodel")

functions = ["linear", "linear_lut", "kmeans"]

for function in functions :
    for bit in [16,8,7,6,5,4,3,2,1]:
        print("processing ",function," on ",bit,".")	
        lin_quant_model = quantize_weights(model, bit, function)
        lin_quant_model.short_description = str(bit)+" bit per quantized weight, using "+function+"."
        lin_quant_model.save(mode_name+"_"+function+"_"+str(bit)+".mlmodel")
        compare_models(model, lin_quant_model, 'testing_data/pizza')


#compare_models(model, lin_quant_model, '/Users/rezashirazian/Projects/Practice/Quantize/testing_data/pizza')

# lin_quant_model.save("/Users/rezashirazian/Projects/Practice/Quantize/food_8_linear.mlmodel")