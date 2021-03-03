import pretty_midi
import numpy as np

def get_comp():
    """
    Drum set that I defined.
    
    Returns
    -------
    standard : matching components to MIDI standard
    encoded : matching components to encoded version
    """
    
    standard = {35: 'kick', 38: 'snare', 
                46: 'open hi-hat', 42: 'closed hi-hat', 
                50: 'high tom', 48: 'mid tom', 45: 'low tom', 
                49: 'crash', 51: 'ride'}
    
    encoded = {'kick': 0, 'snare': 1, 
               'open hi-hat': 2, 'closed hi-hat': 3, 
               'high tom': 4, 'mid tom': 5, 'low tom': 6, 
               'crash': 7, 'ride': 8}
    
    return standard, encoded


def map_unique_drum(note):
    """Matching drum components to closest one.
    I fixed 9 drum components.
     
    Parameters
    ----------
    note : pm.instruments[drum][kick, snare, hi-hat...]

    Returns
    -------
    True : successfully matched
    False : no components to match
    """
    
    pitch = note.pitch
    standard, encoded = get_comp()
    
    # partial mapping
    map_to_standard = {36: 35, # kick 
                       37: 38, 39: 48, 40: 38,# snare
                       44: 42, # closed hi-hat
                       41: 45, 43: 45, # low tom
                       47: 48, # mid tom
                       55: 49, 57: 49, # crash
                       59: 51} #ride
    
    if pitch not in standard.keys():
        if pitch in map_to_standard.keys():
            note.pitch = map_to_standard[pitch]
        else:
            return False
    
    return True


def quantize_drum(inst, fs, start_time, comp=9):
    """Quantize drum sequence according to sampling rate
     
    Parameters
    ----------
    inst : pm.instruments[drum]
    fs : sampling rate that you intended
    start_time : start time of hit drum
    comp : the number of components (default=9)

    Returns
    -------
    drum roll : quantized drum sequence as numpy array
    """
    
    fs_time = 1 / fs
    end_time = inst.get_end_time()
    
    standard, encoded = get_comp()
    
    quantize_time = np.arange(start_time, end_time+fs_time, fs_time)
    drum_roll = np.zeros((quantize_time.shape[0], comp))
    
    for i, note in enumerate(inst.notes):
        # mapping drum to standard set
        if map_unique_drum(note) == False:
            continue
        
        # find nearest index of quantized time
        start_index = np.argmin(np.abs(quantize_time - note.start))
        end_index = np.argmin(np.abs(quantize_time - note.end))
        
        if start_index == end_index:
            end_index += 1
        
        # quantized index
        range_index = np.arange(start_index, end_index)
        
        # choose encoded instruments
        inst_index = encoded[standard[note.pitch]]
        
        # marking where activated
        for index in range_index:
            drum_roll[index, inst_index] = 1
        
    return drum_roll


def drum_play(array, fs, comp=9):
    """Convert numpy object to pretty_midi 
     
    Parameters
    ----------
    array : numpy array (seq, feat)
    fs : sampling rate that you intended
    comp : the number of drum components

    Returns
    -------
    pm : pretty_midi object
    """
        
    fs_time = 1 / fs
    
    standard, encoded = get_comp()
    reverse_standard = {v: k for k, v in standard.items()}
    reverse_encoded = {v: k for k, v in encoded.items()}
    
    decimal_idx= np.where(array == 1)[1]
    binary_idx = list(map(lambda x: np.binary_repr(x, comp), decimal_idx))
    
    # initialize
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=32, is_drum=True)
    pm.instruments.append(inst)
    
    for i, inst_in_click in enumerate(binary_idx):
        start_time = fs_time * i
        end_time = fs_time * (i + 1)
        
        # add instruments
        for j in range(0, len(inst_in_click)):
            if inst_in_click[j] == '1':
                pitch = reverse_standard[reverse_encoded[j]]
                inst.notes.append(pretty_midi.Note(80, pitch, start_time, end_time))
                
    return pm