import enum
import joblib
import numpy as np

general_example_x = np.asarray([[-0.24216784, -0.25672372, -0.03388052, -0.30032498, -0.07865291, -0.52939804,
                                 0.24234088, -0.19052635, -0.26371173, -0.61144378, -1.05309294, -0.49518356]])
general_example_y = 2


class Affinity(enum.Enum):
    GENERAL = 'General'
    CZECH = 'Czech'


class TrainedModel(enum.Enum):
    RANDOM_FOREST = 'RandomForestRegressor'
    GAUSSIAN = 'GaussianNB'
    K_NEAREST = 'KNeighborsClassifier'
    SVC = 'SVC'
    BERNOULLI = 'BernoulliNB'
    DECISION_TREE = 'DecisionTreeClassifier'


def load_model(affinity: Affinity, modelName: TrainedModel = TrainedModel.RANDOM_FOREST):
    return joblib.load(str.join('/', ['models', affinity.value, modelName.value]))


if __name__ == '__main__':
    model = load_model(Affinity.GENERAL)
    prediction = model.predict(general_example_x)

    print('Actual: ', prediction, " and expected: ", general_example_y,
          ' equals.' if prediction == general_example_y else ' different.')
