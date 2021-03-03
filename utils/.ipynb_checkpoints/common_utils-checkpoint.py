import pretty_midi
import numpy as np


def check_time_sign(pm, num=4, denom=4):
    """Check whether the time signature is what you intended.
     
    Parameters
    ----------
    pm : pretty_midi object
    num : numerator of time signature
    denom : denominator of time signature

    Returns
    -------
    Boolean : True of False
    """
        
    time_sign_list = pm.time_signature_changes
    
    # empty check
    if len(time_sign_list) == 0: 
        return False
    
    # nom and denom check
    for time_sign in time_sign_list:
        if time_sign.numerator != num or time_sign.denominator != denom:
            return False
        
    return True


def change_fs(beats, target_beats=16):
    """Get sampling rate converted to what you intended.
    If you want to get time length of sixteenth note, set target_beats=16.
    
     
    Parameters
    ----------
    beats : beats from pm.get_beats()
    target_beats : target beats

    Returns
    -------
    changed_fs : converted sampling rate
    
    Notes
    -----
    - It works on only units of quarter note (4, 8, 16, 32 ...)
    - time_length = 1 / changed_fs
    """
        
    quarter_length = beats[1] - beats[0]
    changed_length = quarter_length /(target_beats / 4)
    changed_fs = 1 / changed_length
    
    return changed_fs


def bin_to_dec(array):
    """Convert numpy binary to decimal.
     
    Parameters
    ----------
    array : numpy array ex) [1, 0, 0, 0, 1, 1]

    Returns
    -------
    decimal : decimal integer value
    """
    
    decimal = 0
    length = array.shape[0]
    
    for i, elem in enumerate(array):
        decimal += (np.power(2, length-i-1) * elem)
        
    return int(decimal)


def hot_encoding(roll):
    """One-hot encoding of n-dims numpy array
    
    Examples
    --------
    intput : (batch, seq, 6) ex) [1, 0, 0, 0, 1, 1]
    output : (batch, seq, 2^6) - one-hot encoded
    
    Parameters
    ----------
    roll : sequence array of which roll[-1] is decimal

    Returns
    -------
    I : one-hot encoded roll
    """
    
    last_axis = len(roll.shape) - 1
    I = np.eye(np.power(2, roll.shape[-1]), dtype='bool') 
    dec_index = np.apply_along_axis(bin_to_dec, last_axis, roll)
    
    return I[dec_index]


def windowing(roll, window_size=64, bar=16, cut_ratio=0.9):
    """Windowing (seq, feat) -> (batch, window, feat)
    If empty degree of a bar is over cut_ratio, discard the window.
    
    Parameters
    ----------
    roll : sequence array (seq, feat)
    window_size : window_size
    bar : the number of units in one bar
    cut_ratio : empty degree of a bar

    Returns
    -------
    new_roll : sequence array (batch, window, feat)
    """
    
    new_roll = []
    num_windows = roll.shape[0] // window_size
    do_nothing = (np.sum((roll == 0), axis=1) == roll.shape[1])
    
    for i in range(0, num_windows):
        break_flag = False
        start_index = window_size * i
        end_index = window_size * (i + 1)
        
        # check empty degree of bars
        check_vacant = do_nothing[start_index:end_index]
        for j in range(0, window_size, bar):
            if np.sum(check_vacant[j:j+bar]) > (bar*cut_ratio):
                break_flag = True
                break
        
        # detected vacant bar
        if break_flag: continue
        new_roll.append(np.expand_dims(roll[start_index:end_index], axis=0))
        
    return np.vstack(new_roll)


def prob_hard_label(prob):
    """
    prob to label with argmax
    
    Parameters
    ----------
    prob : prob sequence (seq, feat)

    Returns
    -------
    play : label sequence (seq, feat)
    """
        
    play = np.zeros(prob.shape)
    label = np.argmax(prob, axis=1)
    
    for seq in range(prob.shape[0]):
        play[seq, label[seq]] = 1
    
    return play


def prob_soft_label(prob):
    """
    prob to label sampled from categorical distribution
    
    Parameters
    ----------
    prob : prob sequence (seq, feat)

    Returns
    -------
    play : label sequence (seq, feat)
    """
        
    num_classes = prob.shape[1]
    play = np.zeros(prob.shape)
    
    for seq in range(prob.shape[0]):
        label = np.random.choice(num_classes, size=1, p=prob[seq])
        play[seq, label] = 1
    
    return play