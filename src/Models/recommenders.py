import numpy as np
import tensorflow_probability as tfp

import src.Models.gBCE
tfd = tfp.distributions
from importlib import reload
import src.Models

class Recommender:
    def __init__(self, hyperparams=None):
        """
        :param hyperparameters: Dictionary containing all relevant hyperparamters, specific to the model at hand.
        """

        pass
    
    def train(self, data_train, data_validation=None, hyperparams=None):
        """
        :param data_train: Trainingsdaten in dem Format, das prepare_data ausgibt.
        :param data_validaiton: Validierungsdatenset in dem Format, das prepare_data ausgibt.
        :param hyperparameters: Dictionary containing all relevant hyperparamters, specific to the model at hand.

        """

        pass

    def predict(self, users, choiceset, rec_size=3):
        """
        :param users: array of user ids
        :param choiceset: array of arrays of item ids 
        :param rec_size: int, number of recommendations per user
        """

        pass

    def cost(self, data, true_prefs):
        pass

    def clear(self):
        # Um zum Beispiel Tensorflow-Sessions zu beenden, sobald das Modell nicht mehr benötigt wird
        pass


class Recommender_multivariate(Recommender):
    """Base class for all recommenders that use a multivariate model."""
    def __init__(self, hyperparams=None):
        self.model = None
        self.hyperparams = hyperparams

    def train(self, data_train, data_validation=None, hyperparams=None):
        data_train = self.model.generate_dataset(
            data_train[:,0],
            data_train[:,1],
            data_train[:,2],
        )
        if data_validation is not None:
            data_validation = self.model.generate_dataset(
                data_validation[:,0],
                data_validation[:,1],
                data_validation[:,2], 
            )

        train_losses, validation_losses = self.model.train_model(
            train_dataset=data_train,
            val_dataset=data_validation,
            learning_rate=self.hyperparams["learning_rate"],
            n_epochs=int(self.hyperparams["n_epochs"]))

        del data_train
        if data_validation is not None:
            del data_validation

        result = {
            "train_losses": train_losses,
            "validation_losses": validation_losses,
        }
        return result

    def predict(self, users, choiceset, rec_size=None):
        preds = []
        for user in users:
            utilities = self.model(np.repeat(user, len(choiceset)), np.asarray(choiceset))
            utilities = np.asarray(utilities).reshape(-1)
            pred_user = [x for _, x in sorted(zip(utilities, choiceset), reverse=True)]
            preds.append(pred_user)

        preds = np.asarray(preds)
        # top rec_size
        if rec_size is not None:
            preds = preds[:, range(rec_size)]
        return(preds)

    def clear(self):
        import tensorflow as tf
        tf.keras.backend.clear_session()


class Recommender_multinomial_logit(Recommender_multivariate):
    """Multinomial Logit Model"""
    def __init__(self, hyperparams=None):
        from src.Models.multinomial_logit import Recommender_Network

        # Run on CPU, because its faster for this small data set size
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')

        self.hyperparams = hyperparams
        self.model = Recommender_Network(
            n_users=int(hyperparams["n_users"]),  
            n_items=int(hyperparams["n_alternatives"]),  
            embedding_size=int(hyperparams["k"]),  
            batch_size=int(hyperparams["batch_size"]),  
            l2_embs = hyperparams["l2_embs"],
            n_early_stop = hyperparams["n_early_stop"],
            optimizer_class_name = hyperparams["optimizer_class_name"],
        )

    def train(self, data_train, data_validation=None, hyperparams=None):
        # reload(src.Models.multinomial_logit)
        # import src.Models.multinomial_logit
        return super().train(data_train, data_validation, hyperparams)


class Recommender_exponomial(Recommender_multivariate):
    """Exponomial Model"""
    def __init__(self, hyperparams=None):
        from src.Models.exponomial import Recommender_Network

        # Run on CPU, because its faster for this small data set size
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')

        self.hyperparams = hyperparams
        self.model = Recommender_Network(
            n_users=int(hyperparams["n_users"]),  
            n_items=int(hyperparams["n_alternatives"]),  
            embedding_size=int(hyperparams["k"]), 
            batch_size=int(hyperparams["batch_size"]),  
            l2_embs = hyperparams["l2_embs"],
            n_early_stop = hyperparams["n_early_stop"],
            optimizer_class_name = hyperparams["optimizer_class_name"],
        )

    def train(self, data_train, data_validation=None, hyperparams=None):
        # reload(src.Models.exponomial)
        return super().train(data_train, data_validation, hyperparams)


# Neural MNL Matrix Factorization
class Recommender_NPMLE(Recommender_multivariate):
    def __init__(self, hyperparams=None):
        from src.Models.npmle import Recommender_Network

        # Run on CPU, because its faster for this small data set size
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')

        self.hyperparams = hyperparams
        self.model = Recommender_Network(
            n_users=int(hyperparams["n_users"]),  
            n_items=int(hyperparams["n_alternatives"]),  
            embedding_size=int(hyperparams["k"]),  
            n_kernels=hyperparams["n_kernels"],
            batch_size=int(hyperparams["batch_size"]),  
            l2_embs = hyperparams["l2_embs"],
            n_early_stop=int(hyperparams["n_early_stop"]), 
            normalize_bias=hyperparams["normalize_bias"],
            learn_bandwidth=hyperparams["learn_bandwidth"],
            plot_cdf_gif=hyperparams.get("plot_cdf_gif", True),
            optimizer_class_name = hyperparams["optimizer_class_name"],
        )

    def train(self, data_train, data_validation=None, hyperparams=None):
        reload(src.Models.npmle)
        from src.Models.npmle import Recommender_Network

        if hyperparams is not None:
            self.hyperparams = hyperparams
            
        # Daten vorverarbeiten
        data_train = self.model.generate_dataset(
            data_train[:,0],
            data_train[:,1],
            data_train[:,2],
        )

        if data_validation is not None:
            data_validation = self.model.generate_dataset(
                data_validation[:,0],
                data_validation[:,1],
                data_validation[:,2],
            )

        train_losses, validation_losses = self.model.train_model(
            train_dataset=data_train,
            val_dataset=data_validation,
            learning_rate_model=self.hyperparams["learning_rate_model"],
            learning_rate_cdf=self.hyperparams["learning_rate_cdf"],
            n_epochs=int(self.hyperparams["n_epochs"]),
        )
        
        del data_train
        if data_validation is not None:
            del data_validation

        result = {
            "train_losses": train_losses,
            "validation_losses": validation_losses,
        }
        return result


class Recommender_binary_logit(Recommender_multivariate):
    """Binary Logit Model (Matrix Factorization)"""
    def __init__(self, hyperparams=None):
        from src.Models.binary_logit import Recommender_Network
        self.hyperparams = hyperparams

        # Run on CPU, because its faster for this small data set size
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')

        self.model = Recommender_Network(
            n_users=int(hyperparams["n_users"]),  
            n_items=int(hyperparams["n_alternatives"]),  
            embedding_size=int(hyperparams["k"]),  
            batch_size=int(hyperparams["batch_size"]),  
            l2_embs = hyperparams["l2_embs"],
            n_early_stop=int(hyperparams["n_early_stop"]), 
            optimizer_class_name = hyperparams["optimizer_class_name"],
        )


    def get_negative_interactions(self, data_train, data_validation=None):
        # Positive interactions
        data_train_pos = np.asarray([data_train[:,0], data_train[:,2], np.ones(len(data_train[:,0]))]).transpose()
        if data_validation is not None:
            data_validation_pos = np.asarray([data_validation[:,0], data_validation[:,2], np.ones(len(data_validation[:,0]))]).transpose()
        else: 
            data_validation_pos = None

        data_train_neg = []
        for i in range(len(data_train)):
            user = data_train[i,0]
            choice = data_train[i,2]
            for option in data_train[i,1]:
                if option != choice:
                    data_train_neg.append([user, option, 0])
        data_train_neg = np.asarray(data_train_neg)

        data_train = np.concatenate([data_train_pos, data_train_neg])

        data_train = data_train.astype("int32")

        if data_validation is not None:
            data_validation_neg = []
            for i in range(len(data_validation)):
                user = data_validation[i,0]
                choice = data_validation[i,2]
                for option in data_validation[i,1]:
                    if option != choice:
                        data_validation_neg.append([user, option, 0])
            data_validation_neg = np.asarray(data_validation_neg)

            data_validation = np.concatenate([data_validation_pos, data_validation_neg])

            data_validation = data_validation.astype("int32")

        return data_train, data_validation


    def train(self, data_train, data_validation=None, hyperparams=None):
        # reload(src.Models.binary_logit)

        data_train_sampled, data_validation_sampled = self.get_negative_interactions(data_train, data_validation)

        data_train = self.model.generate_dataset(data_train_sampled)
        if data_validation is not None:
            data_validation = self.model.generate_dataset(data_validation_sampled)

        train_losses, validation_losses = self.model.train_model(
            train_dataset=data_train,
            val_dataset=data_validation,
            learning_rate=self.hyperparams["learning_rate"],
            n_epochs=int(self.hyperparams["n_epochs"])
        )

        del data_train
        if data_validation is not None:
            del data_validation
            
        result = {
            "train_losses": train_losses,
            "validation_losses": validation_losses,
        }
        
        return result

class Recommender_binary_logit_negative_sampling(Recommender_binary_logit):
    """Binary Logit Model (Matrix Factorization) with negative sampling"""
    def __init__(self, hyperparams=None):
        from src.Models.binary_logit_negative_sampling import Recommender_Network
        self.hyperparams = hyperparams

        # Run on CPU, because its faster for this small data set size
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        
        self.model = Recommender_Network(
            n_users=int(hyperparams["n_users"]),  
            n_items=int(hyperparams["n_alternatives"]),  
            embedding_size=int(hyperparams["k"]),  
            batch_size=int(hyperparams["batch_size"]), 
            l2_embs = hyperparams["l2_embs"],
            n_early_stop=int(hyperparams["n_early_stop"]), 
            optimizer_class_name = hyperparams["optimizer_class_name"],
        )


    def train(self, data_train, data_validation=None, hyperparams=None):
        # reload(src.Models.binary_logit_negative_sampling)

        # Daten vorverarbeiten
        train_losses, validation_losses = self.model.train_model(
            data_train=data_train,
            data_val=data_validation,
            learning_rate=self.hyperparams["learning_rate"],
            n_epochs=int(self.hyperparams["n_epochs"]),
        )

        del data_train
        if data_validation is not None:
            del data_validation

        result = {
            "train_losses": train_losses,
            "validation_losses": validation_losses,
        }
        return result


class Recommender_gBCE(Recommender_binary_logit_negative_sampling):
    """gBCE from https://dl.acm.org/doi/10.1145/3604915.3608783"""
    def __init__(self, hyperparams=None):
        from src.Models.gBCE import Recommender_Network
        self.hyperparams = hyperparams

        # Run on CPU, because its faster for this small data set size
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        
        self.model = Recommender_Network(
            n_users=int(hyperparams["n_users"]),  
            n_items=int(hyperparams["n_alternatives"]),  
            embedding_size=int(hyperparams["k"]),  
            batch_size=int(hyperparams["batch_size"]), 
            l2_embs = hyperparams["l2_embs"],
            n_early_stop=int(hyperparams["n_early_stop"]), 
            t=hyperparams["t"],
            optimizer_class_name = hyperparams["optimizer_class_name"],
        )

    def train(self, data_train, data_validation=None, hyperparams=None):
        # reload(src.Models.gBCE)
        return super().train(data_train, data_validation, hyperparams)

