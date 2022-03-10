from models.neural import LightningNeuralNetModel
from models.pytorch.cnn import ConvolutionalModel
from models.pytorch.mlp import MultiLayerPerceptron
from data.data_loader import load_data
from embedding import embed_grid
from embedding.program_discretized import DiscretizedProgramStaticEmbeddings
from utils.parse import parse_grid, parse_program
from config import ProgramSynthesisTask, dataconfig_basic
from utils.scoring import score_programs
from render.visualize import visualize
from pytorch_lightning import seed_everything
import pandas as pd

seed_everything(42, workers=True)

dataconfig = dataconfig_basic
program_embedding = DiscretizedProgramStaticEmbeddings(dataconfig)

mlp_model = MultiLayerPerceptron(
    hidden_layers_ratio=[1.0, 2.0],
    dropout_ratio=0.1,
    loss_function=program_embedding.loss,
)
conv_model = ConvolutionalModel(
    loss_function=program_embedding.loss,
    dropout_p=0.2,
)


task = ProgramSynthesisTask(
    data_loader=load_data,
    parse_input=parse_grid(dataconfig),
    parse_program=parse_program,
    embed_input=embed_grid(dataconfig),
    program_embedding=program_embedding,
    scorer=score_programs,
    model=LightningNeuralNetModel(conv_model, max_epochs=100, logging=False),
    visualize=visualize(dataconfig),
    dataset_name=dataconfig.name,
)


def run_pipeline(task: ProgramSynthesisTask):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(task.dataset_name)

    X_train = [task.embed_input(task.parse_input(row)) for row in X_train]
    y_train = [
        task.program_embedding.program_to_tensor(task.parse_program(row))
        for row in y_train
    ]
    X_val = [task.embed_input(task.parse_input(row)) for row in X_val]
    y_val = [
        task.program_embedding.program_to_tensor(task.parse_program(row))
        for row in y_val
    ]
    X_test_without_embedding = [task.parse_input(row) for row in X_test]
    X_test = [task.embed_input(row) for row in X_test_without_embedding]
    y_test = [task.parse_program(row) for row in y_test]

    result = task.model.fit(X_train, y_train, X_val, y_val)
    output = task.model.predict(X_test)
    output_programs = task.program_embedding.tensor_to_programs(output)

    score = score_programs(y_test, output_programs)
    print(f"Score: {score}")
    if task.visualize is not None:
        task.visualize(X_test_without_embedding, output_programs)

    pd.Series(result).to_csv("output/results.csv")


if __name__ == "__main__":
    run_pipeline(task)
