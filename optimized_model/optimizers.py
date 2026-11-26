import numpy as np

#NEED TO ADD CODE FOR HUFFMAN OPTIMIZATION

class Optimizer:
    """Base class — shared parameter storage and zero_grad."""

    def __init__(self, parameters):
        self.parameters = list(parameters)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

    def step(self):
        raise NotImplementedError

class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation), Kingma & Ba (2014).

    Maintains exponential moving averages of the gradient (first moment,
    `m`) and the squared gradient (second moment, `v`), with bias
    correction for their initialization at zero:

        m <- beta1 * m + (1 - beta1) * grad
        v <- beta2 * v + (1 - beta2) * grad^2
        m_hat <- m / (1 - beta1^t)
        v_hat <- v / (1 - beta2^t)
        p.data <- p.data - lr * m_hat / (sqrt(v_hat) + eps)

    Adam adapts the effective learning rate per-parameter, which generally
    makes it a robust default for training CNNs without much learning-rate
    tuning.
    """

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in self.parameters]
        self._v = [np.zeros_like(p.data) for p in self.parameters]
        self._t = 0

    def step(self):
        self._t += 1
        for p, m, v in zip(self.parameters, self._m, self._v):
            m *= self.beta1
            m += (1 - self.beta1) * p.grad
            v *= self.beta2
            v += (1 - self.beta2) * (p.grad ** 2)

            m_hat = m / (1 - self.beta1 ** self._t)
            v_hat = v / (1 - self.beta2 ** self._t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class HuffmanEncoding(Optimizer):
    def __init__(self, model_path, test_data, train_data, k, lr):
        self.test_data = test_data
        self.train_data = train_data
        self.model = torch.load(model_path)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.k = k
        self.lr = lr


    def train(self, optimizer=None, epoches=10):
        self.model = self.model.cuda()
        if optimizer is None:
            optimizer = \
                optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)


        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch(optimizer, weight_share=True)
        print("Finished fine tuning.")
        return self.update_weights()

    def update_weights(self):
        model = self.model
        for layer, (name, module) in enumerate(model.classifier._modules.items()):
            module.register_backward_hook(self.scalar_quantization)
            weight = module.weight.data.cpu().numpy()
            weight_shape = weight.shape
            centroids = module.centroids
            labels = module.labeled_weight
            new_weight = self.get_finilized_weight(weight=weight, centroids=centroids, labels=labels)
            new_weight = new_weight.reshape(weight_shape[0], weight_shape[1], dtype=np.int8)
            module.weight = torch.from_numpy(new_weight).cuda().int()
            del module.labeled_weight
        return model

    def get_finilized_weight(self, weight, centroids, labels):
        for index, label in enumerate(labels):
            weight[index] = centroids[label][0]
        return weight


    def train_batch(self, optimizer, batch, label, weight_share):
        self.model.zero_grad()
        input = Variable(batch)
        if weight_share:
            output = self.forward(input)
            self.criterion(output, Variable(label)).backward()
        else:
            self.criterion(self.model(input), Variable(label)).backward()
            optimizer.step()

    def train_epoch(self, optimizer=None, weight_share=False):
        index = 1
        for batch_index, (batch, label) in enumerate(self.train_data, 0):
            self.train_batch(optimizer, batch.cuda(), label.cuda(), weight_share)
            if batch_index % 100 == 0:
                print(batch_index)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), -1)
        for layer, (name, module) in enumerate(self.model.classifier._modules.items()):
            if isinstance(module, torch.nn.modules.Linear):
                module.register_backward_hook(self.scalar_quantization)
                weight = module.weight.data.cpu().numpy()
                weight_shape = weight.shape
                sorted_centroids, centroids, labeled_weight = self.find_centroids(weight, self.k)
                new_weight = self.get_converted_weight(labeled_weight=labeled_weight, centroids=centroids)
                new_weight = new_weight.reshape(weight_shape[0], weight_shape[1])
                module.labeled_weight = labeled_weight
                module.centroids = centroids
                module.weight.data = torch.from_numpy(new_weight).float().cuda()
            x = module(x)
        return x

    def find_centroids(self, weight, num_class):
        a = weight.reshape(-1, 1)
        kmeans = KMeans(n_clusters=num_class, random_state=0).fit(a)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        sorted_centroids = -np.sort(-centroids, axis=0)
        return sorted_centroids, centroids, labels

    def get_converted_weight(self, labeled_weight, centroids):
        new_weight = np.zeros(shape=labeled_weight.shape, dtype=np.float32)
        for index, label in enumerate(labeled_weight):
            new_weight[index] = centroids[label][0]
        return labeled_weight

    def get_centroids_gradients(self, grad_input, labeled_weight, dw, grad_output):
        w_grad = grad_input[2].t().data.cpu().numpy()
        grad_w = w_grad.reshape(-1, 1)
        for index, label in enumerate(labeled_weight):
            dw[label][0] += grad_w[index]
        return dw

    def scalar_quantization(self, module, grad_input, grad_output):
        if isinstance(module, nn.Linear):
            labeled_weight = module.labeled_weight
            centroids = module.centroids
            dw = np.zeros(shape=centroids.shape, dtype=np.float32)
            dw = self.get_centroids_gradients(grad_input, labeled_weight, dw, grad_output)
            module.centroids = centroids - (self.lr * dw)
