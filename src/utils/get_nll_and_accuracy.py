from src.Models.recommenders import Recommender_binary_logit, Recommender_binary_logit_negative_sampling, Recommender_gBCE

def get_nll_and_accuracy(model, model_class, data_test, users_test, options_test, choices_pos_test):
    if model_class in [Recommender_binary_logit_negative_sampling, Recommender_gBCE]:
        nll, acc = model.model.val_step(
            *(model.model.sample_negatives(users=data_test[:,0], validation=True).T)
        )
    elif model_class == Recommender_binary_logit:
        data_test_binary = model.get_negative_interactions(data_test)[0]
        nll, acc = model.model.val_step(
            data_test_binary[:, 0], 
            data_test_binary[:, 1], 
            data_test_binary[:, 2],
        )
    else:
        nll, acc = model.model.val_step(
            users_test, options_test, choices_pos_test
        )

    return nll, acc
