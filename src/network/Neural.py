import Layer

class Neural:
    def __init__(self, layer_size: list):
        self.layers = []
        

        
        
        for i in range(len(layer_size) - 1):
            
            self.layers.append(Layer.Layer(layer_size[i], layer_size[i+1]))
        
        if len(self.layers) != len(layer_size) - 1:
            print("Error")
            return
        
    
    def calculate_outputs(self, inputs: list):
        for i in self.layers:
            inputs = i.calculate_outputs(inputs)
            
        return inputs
    
    
    def classify(self, inputs: list):
        outputs = self.calculate_outputs(inputs)
        
        outputs.sort(reverse=True)
        
        return outputs[0]

    def loss(self, data_point):
        outputs = self.calculate_outputs(data_point.inputs)
        
        output_layer = self.layers[len(self.layers) - 1]
        
        loss = 0
        
        for i in range(len(outputs)):
            loss += output_layer.cost(outputs[i], data_point.expected_outputs[i])
        
        return loss
    
    def learn(self, training_data, learn_rate):
        h = 0.0001
        
        original_loss = self.loss(training_data)
        
        for i in self.layers:
            for j in range(self.layers[i].num_of_nodes_in):
                
                for k in range(self.layers[i].num_of_nodes_out):
                    
                    self.layers[i].weights[j][k] += h
                    
                    delta_loss = self.loss(training_data) - original_loss
                    
                    self.layers[i].weights[j][k] -= h
                    
                    self.layers[i].loss_gradient_w[j][k] = delta_loss / h
            
            for l in range(len(self.layers[i].biases)):
                self.layers[i].biases[l] += h
                
                delta_loss = self.loss(training_data) - original_loss
                
                self.layers[i].biases[l] -= h
                
                self.layers[i].loss_gradient_b[l] = delta_loss / h
                
            self.layers[i].apply_gradient(learn_rate)

    def update_all_gradients(self, data_point):
        self.calculate_outputs(data_point)
        
        output_layer = self.layers[len(self.layers) - 1]
        
        node_values = output_layer.calculate_output_layer_node_values(data_point.expectedOutputs)