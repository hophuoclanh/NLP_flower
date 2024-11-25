from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig, ServerApp
from flwr.common import ndarrays_to_parameters
from transformers import AutoModelForSequenceClassification

def create_server(model_name, num_labels, num_rounds):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    initial_parameters = ndarrays_to_parameters(get_weights(model))

    strategy = FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_available_clients=2,
        initial_parameters=initial_parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerApp(strategy=strategy, config=config)
