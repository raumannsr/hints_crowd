from keras.models import model_from_json
from get_data import get_data_2
from generate_data import generate_data_2
from report_results import report_auc
from sklearn.metrics import roc_auc_score
from get_data import annototation_type
import numpy as np
from scipy import optimize
import enum
import keras

class ensemble_learning_type(enum.Enum):
   soft_voting = 1
   averaging = 2
   optimizing = 3

# RUN FLAGS
VERBOSE = True
SANITY_CHECK = False
BOROMIR = True
CONV_LAYER_FROZEN = True
ENSEMBLE_LEARNING = ensemble_learning_type.optimizing

# DEFINITIONS
GROUP_PATH = '../../0_data/student_2018_abc_features/data/'
REPORT_PATH = '../reports/'
WEIGHTS_PATH = '../weights/'
TRUTH_CSV = 'ISIC-2017_Training_Part3_GroundTruth.csv'
BATCH_SIZE = 20
TRUTH_PATH = '../../0_data/student_2018_abc_features/data/'

if BOROMIR:
    IMAGE_DATA_PATH = '/data/ralf/19/'
else:
    IMAGE_DATA_PATH = '/data/CrowdSkin/ekcontar/dat/'

def loss_mse(weights, valid_labels, pred_val):
    predictions = np.average(pred_val, weights=weights, axis=0)
    mse = keras.losses.MeanSquaredError()
    loss = mse(valid_labels, predictions).numpy()
    return loss

def reconstruct_model(weights_path):
    json_file = open(weights_path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path + '.h5')
    if VERBOSE:
        print('MODEL path = ', str(weights_path))
        model.summary()
    return model

def validate_model(model):
    valid = generate_data_2(directory=IMAGE_DATA_PATH,
                           augmentation=False,
                           batch_size=BATCH_SIZE,
                           file_list=valid_id,
                           label_1=valid_label_c,
                           label_2=valid_label_a,
                           sample_weights = valid_mask)
    predictions = model.predict_generator(valid, 25)
    delta_size = predictions[0].size - valid_label_c.count()
    scores = np.resize(predictions[0], predictions[0].size - delta_size)
    return scores

def predict_model(model):
    test = generate_data_2(directory=IMAGE_DATA_PATH,
                           augmentation=False,
                           batch_size=BATCH_SIZE,
                           file_list=test_id,
                           label_1=test_label_c,
                           label_2=test_label_a,
                           sample_weights = test_mask)
    predictions = model.predict_generator(test, 25)
    delta_size = predictions[0].size - test_label_c.count()
    scores=np.resize(predictions[0], predictions[0].size - delta_size)
    return scores

seeds = [1970, 1972, 2008, 2019, 2020]
for seed in seeds:
    pred_validation = np.zeros([3, 350])
    pred_test = np.zeros([3,250])

    if VERBOSE:
        print('*************************************************************')
        print('Seed              = ', str(seed))
        print('Frozen conv_base  = ', str(CONV_LAYER_FROZEN))
        print('Sanity check      = ', str(SANITY_CHECK))
        print('Ensemble learning = ', str(ENSEMBLE_LEARNING))
        print('*************************************************************')

    if CONV_LAYER_FROZEN:
        asy_model = reconstruct_model(WEIGHTS_PATH + 'model_frozen' + str(seed) + str(annototation_type.asymmetry))
        bor_model = reconstruct_model(WEIGHTS_PATH + 'model_frozen' + str(seed) + str(annototation_type.border))
        col_model = reconstruct_model(WEIGHTS_PATH + 'model_frozen' + str(seed) + str(annototation_type.color))
    else:
        asy_model = reconstruct_model(WEIGHTS_PATH + 'model_fully_train' + str(seed) + str(annototation_type.asymmetry))
        bor_model = reconstruct_model(WEIGHTS_PATH + 'model_fully_train' + str(seed) + str(annototation_type.border))
        col_model = reconstruct_model(WEIGHTS_PATH + 'model_fully_train' + str(seed) + str(annototation_type.color))

    train_id, valid_id, test_id, train_label_c, valid_label_c, test_label_c, train_label_a, valid_label_a, test_label_a, train_mask, valid_mask, test_mask, class_weights = get_data_2(
            GROUP_PATH, TRUTH_PATH, TRUTH_CSV, seed, VERBOSE, SANITY_CHECK, annototation_type.asymmetry)
    asy_pred_score = predict_model(asy_model)
    if ENSEMBLE_LEARNING == ensemble_learning_type.optimizing:
        asy_val_score = validate_model(asy_model)
        pred_validation[0, :] = asy_val_score
        pred_test[0, :] = asy_pred_score

    train_id, valid_id, test_id, train_label_c, valid_label_c, test_label_c, train_label_a, valid_label_a, test_label_a, train_mask, valid_mask, test_mask, class_weights = get_data_2(
            GROUP_PATH, TRUTH_PATH, TRUTH_CSV, seed, VERBOSE, SANITY_CHECK, annototation_type.border)
    bor_pred_score = predict_model(bor_model)
    if ENSEMBLE_LEARNING == ensemble_learning_type.optimizing:
        bor_val_sore = validate_model(bor_model)
        pred_validation[1, :] = bor_val_sore
        pred_test[1, :] = bor_pred_score

    train_id, valid_id, test_id, train_label_c, valid_label_c, test_label_c, train_label_a, valid_label_a, test_label_a, train_mask, valid_mask, test_mask, class_weights = get_data_2(
            GROUP_PATH, TRUTH_PATH, TRUTH_CSV, seed, VERBOSE, SANITY_CHECK, annototation_type.color)
    col_pred_score = predict_model(col_model)
    if ENSEMBLE_LEARNING == ensemble_learning_type.optimizing:
        cor_val_score = validate_model(col_model)
        pred_validation[2, :] = cor_val_score
        pred_test[2, :] = col_pred_score

    if ENSEMBLE_LEARNING == ensemble_learning_type.optimizing:
        weights = np.array([0.33,0.33,0.33])
        weights_optim = optimize.differential_evolution(loss_mse,
                                                        bounds=[(0.0, 1.0) for _ in range(len(weights))],
                                                        args=(valid_label_c, pred_validation),
                                                        maxiter=1000,
                                                        tol=1e-7)
        if VERBOSE:
            print(weights_optim.x)
        scores = np.average(pred_test, weights=weights_optim.x, axis=0)
    else:
        scores = []
        for i in range(0, len(test_id)):
            if ENSEMBLE_LEARNING == ensemble_learning_type.soft_voting:
                max_prob = asy_pred_score[i]
                if bor_pred_score[i] > max_prob:
                    max_prob = bor_pred_score[i]
                if col_pred_score[i] > max_prob:
                    max_prob = col_pred_score[i]
                scores.append(max_prob)
            else:
                if ENSEMBLE_LEARNING == ensemble_learning_type.averaging:
                    scores.append((asy_pred_score[i] + bor_pred_score[i] + col_pred_score[i]) / 3.0)

    aucs = []
    auc = roc_auc_score(test_label_c, scores)
    aucs.append(auc)

    if CONV_LAYER_FROZEN:
        filename = REPORT_PATH + 'aucs_frozen' + str(seed) + '.csv'
    else:
        filename = REPORT_PATH + 'aucs_fully_train' + str(seed) + '.csv'

    report_auc(aucs, filename)