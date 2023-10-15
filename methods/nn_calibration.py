import numpy as np
import torch
from scipy.optimize import minimize 
from sklearn.metrics import log_loss
import time
from sklearn.metrics import log_loss, brier_score_loss
from os.path import join
from betacal import BetaCalibration
import sklearn.metrics as metrics
from sklearn.isotonic import IsotonicRegression

# Adapted from open-source code
# https://github.com/markus93/NN_calibration/blob/master/scripts/calibration/cal_methods.py
# https://github.com/dirichletcal/experiments_dnn/blob/master/scripts/calibration/cal_methods.py

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    
    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)
    

class HistogramBinning():
    """
    Histogram Binning as a calibration method. The bins are divided into equal lengths.
    
    The class contains two methods:
        - fit(probs, true), that should be used with validation data to train the calibration model.
        - predict(probs), this method is used to calibrate the confidences.
    """
    
    def __init__(self, M=15):
        """
        M (int): the number of equal-length bins used
        """
        self.bin_size = 1./M  # Calculate bin size
        self.conf = []  # Initiate confidence list
        self.upper_bounds = np.arange(self.bin_size, 1+self.bin_size, self.bin_size)  # Set bin bounds for intervals

    
    def _get_conf(self, conf_thresh_lower, conf_thresh_upper, probs, true):
        """
        Inner method to calculate optimal confidence for certain probability range
        
        Params:
            - conf_thresh_lower (float): start of the interval (not included)
            - conf_thresh_upper (float): end of the interval (included)
            - probs : list of probabilities.
            - true : list with true labels, where 1 is positive class and 0 is negative).
        """

        # Filter labels within probability range
        filtered = [x[0] for x in zip(true, probs) if x[1] > conf_thresh_lower and x[1] <= conf_thresh_upper]
        nr_elems = len(filtered)  # Number of elements in the list.

        if nr_elems < 1:
            return 0
        else:
            # In essence the confidence equals to the average accuracy of a bin
            conf = sum(filtered)/nr_elems  # Sums positive classes
            return conf
    

    def fit(self, probs, true):
        """
        Fit the calibration model, finding optimal confidences for all the bins.
        
        Params:
            probs: probabilities of data
            true: true labels of data
        """

        conf = []

        # Got through intervals and add confidence to list
        for conf_thresh in self.upper_bounds:
            temp_conf = self._get_conf((conf_thresh - self.bin_size), conf_thresh, probs = probs, true = true)
            conf.append(temp_conf)

        self.conf = conf


    # Fit based on predicted confidence
    def predict(self, probs):
        """
        Calibrate the confidences
        
        Param:
            probs: probabilities of the data (shape [samples, classes])
            
        Returns:
            Calibrated probabilities (shape [samples, classes])
        """

        # Go through all the probs and check what confidence is suitable for it.
        for i, prob in enumerate(probs):
            idx = np.searchsorted(self.upper_bounds, prob)
            probs[i] = self.conf[idx]

        return probs
        
        
class TemperatureScaling():
    
    def __init__(self, temp = 1, maxiter = 50, solver = "BFGS"):
        """
        Initialize class
        
        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver
    
    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict(probs, x)    
        loss = log_loss(y_true=true, y_pred=scaled_probs)
        return loss
    
    # Find the temperature
    def fit(self, logits, true):
        """
        Trains the model and finds optimal temperature
        
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.
            
        Returns:
            the results of optimizer after minimizing is finished.
        """
        
        true = true.flatten() # Flatten y_val
        opt = minimize(self._loss_fun, x0 = 1, args=(logits, true), options={'maxiter':self.maxiter}, method = self.solver)
        self.temp = opt.x[0]
        
        return opt
        
    def predict(self, logits, temp = None):
        """
        Scales logits based on the temperature and returns calibrated probabilities
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        if not temp:
            return softmax(logits/self.temp)
        else:
            return softmax(logits/temp)
            
def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    # Computes accuracy and average confidence for bin
    
    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
    
    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct)/len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin
  

def ECE(conf, pred, true, bin_size = 0.1):

    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        ece: expected calibration error
    """
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins
    
    n = len(conf)
    ece = 0  # Starting error
    
    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)        
        ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE
        
    return ece
        
      
def MCE(conf, pred, true, bin_size = 0.1):

    """
    Maximal Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        mce: maximum calibration error
    """
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    
    cal_errors = []
    
    for conf_thresh in upper_bounds:
        acc, avg_conf, _ = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)
        cal_errors.append(np.abs(acc-avg_conf))
        
    return max(cal_errors)
    

def get_bin_info(conf, pred, true, bin_size = 0.1):

    """
    Get accuracy, confidence and elements in bin information for all the bins.
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        (acc, conf, len_bins): tuple containing all the necessary info for reliability diagrams.
    """
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    
    accuracies = []
    confidences = []
    bin_lengths = []
    
    for conf_thresh in upper_bounds:
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)
        accuracies.append(acc)
        confidences.append(avg_conf)
        bin_lengths.append(len_bin)
        
        
    return accuracies, confidences, bin_lengths

def evaluate(probs, y_true, verbose = False, normalize = False, bins = 15):
    """
    Evaluate model using various scoring measures: Error Rate, ECE, MCE, NLL, Brier Score
    
    Params:
        probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: a list containing the actual class labels
        verbose: (bool) are the scores printed out. (default = False)
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized.
        bins: (int) - into how many bins are probabilities divided (default = 15)
        
    Returns:
        (error, ece, mce, loss, brier), returns various scoring measures
    """
    
    preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction
    
    if normalize:
        confs = np.max(probs, axis=1)/np.sum(probs, axis=1)
        # Check if everything below or equal to 1?
    else:
        confs = np.max(probs, axis=1)  # Take only maximum confidence
    
    accuracy = metrics.accuracy_score(y_true, preds) * 100
    error = 100 - accuracy
    
        # Calculate ECE
    ece = ECE(confs, preds, y_true, bin_size = 1/bins)
    # Calculate MCE
    mce = MCE(confs, preds, y_true, bin_size = 1/bins)
    
    loss = log_loss(y_true=y_true, y_pred=probs)
    
    y_prob_true = np.array([probs[i, idx] for i, idx in enumerate(y_true)])  # Probability of positive class
    brier = brier_score_loss(y_true=y_true, y_prob=y_prob_true)  # Brier Score (MSE)
    
    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print("ECE:", ece)
        print("MCE:", mce)
        print("Loss:", loss)
        print("brier:", brier)
    
    return (error, ece, mce, loss, brier)
    
 
def cal_results(fn, 
                train_logits,
                train_labels,
                test_logits,
                m_kwargs={}, 
                approach="1-vs-k"):
    
    """
    Calibrate models scores, using output from logits files and given function (fn). 
    There are implemented to different approaches "all" and "1-vs-K" for calibration,
    the approach of calibration should match with function used for calibration.
    
    TODO: split calibration of single and all into separate functions for more use cases.
    
    Params:
        fn (class): class of the calibration method used. It must contain methods "fit" and "predict", 
                    where first fits the models and second outputs calibrated probabilities.
        path (string): path to the folder with logits files
        files (list of strings): pickled logits files ((logits_val, y_val), (logits_test, y_test))
        m_kwargs (dictionary): keyword arguments for the calibration class initialization
        approach (string): "all" for multiclass calibration and "1-vs-K" for 1-vs-K approach.
        
    Returns:
        df (pandas.DataFrame): dataframe with calibrated and uncalibrated results for all the input files.
    
    """
    y_val = train_labels
    logits_val = train_logits
    logits_test = test_logits
    
    if approach == "all":            

        y_val = y_val.flatten()

        model = fn(**m_kwargs)

        model.fit(logits_val, y_val)

        probs_val = model.predict(logits_val) 
        probs_test = model.predict(logits_test)
        return probs_test
        
    else:  # 1-vs-k models
        probs_val = softmax(logits_val)  # Softmax logits
        probs_test = softmax(logits_test)
        K = probs_test.shape[1]
        
        # Go through all the classes
        for k in range(K):
            # Prep class labels (1 fixed true class, 0 other classes)
            y_cal = np.array(y_val == k, dtype="int")
            
            # print("sum y_cal ", np.sum(y_cal))
            if np.sum(y_cal) < 1:
                probs_test[:, k] = 0
                continue

            # Train model
            model = fn(**m_kwargs)
            model.fit(probs_val[:, k], y_cal) # Get only one column with probs for given class "k"

            probs_val[:, k] = model.predict(probs_val[:, k])  # Predict new values based on the fitting
            probs_test[:, k] = model.predict(probs_test[:, k])

            # Replace NaN with 0, as it should be close to zero  # TODO is it needed?
            idx_nan = np.where(np.isnan(probs_test))
            probs_test[idx_nan] = 0

            idx_nan = np.where(np.isnan(probs_val))
            probs_val[idx_nan] = 0

    probs_test += 1e-10 # avoid 1.0 and 0.0
    probs_test = probs_test / np.sum(probs_test, axis=1, keepdims=True)
    return probs_test
    

def calibrate(
    method_name,
    train_logits,
    train_labels,
    test_logits,
    *args, **kwargs):
    train_logits = train_logits.numpy()
    train_labels = train_labels.numpy()
    test_logits = test_logits.numpy()
    if method_name == "temp_scaling_ref":
        calibrated_prob = cal_results(fn=TemperatureScaling,
                                    train_logits=train_logits,
                                    train_labels=train_labels,
                                    test_logits=test_logits,
                                    approach="all")
    elif method_name == "histogram_binning":
        calibrated_prob = cal_results(fn=HistogramBinning,
                                    train_logits=train_logits,
                                    train_labels=train_labels,
                                    test_logits=test_logits,
                                    approach="1-vs-k")
    elif method_name == "beta_calibration":
        calibrated_prob = cal_results(fn=BetaCalibration, 
                                      train_logits=train_logits,
                                      train_labels=train_labels,
                                      test_logits=test_logits, 
                                      m_kwargs={'parameters':"abm"}, 
                                      approach="1-vs-k")
    elif method_name == "isotonic_regression":
        calibrated_prob = cal_results(fn=IsotonicRegression, 
                                      train_logits=train_logits,
                                      train_labels=train_labels,
                                      test_logits=test_logits, 
                                      m_kwargs={'y_min':0, 'y_max':1}, 
                                      approach="1-vs-k")
    else:
        raise ValueError(method_name)
    return {
        "prob": torch.from_numpy(calibrated_prob)
    }