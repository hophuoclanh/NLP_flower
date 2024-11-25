from flwr.client import NumPyClient

class NewsCategoryClient(NumPyClient):
    def __init__(self, model, trainloader, testloader, local_epochs):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        train(self.model, self.trainloader, epochs=self.local_epochs, device=self.device)
        return get_weights(self.model), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        loss, accuracy = test(self.model, self.testloader, self.device)
        return loss, len(self.testloader), {"accuracy": accuracy}
