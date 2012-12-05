import numpy as np


def all_close(np_array1, np_array2, labels=None):
    if labels is None:
        labels = ['Array %d' % i for i in range(2)]
    else:
        labels = list(labels[:])
    labels.append('Diff')

    label_lengths = [len(l) for l in labels]
    max_label_length = max(label_lengths)


    message_template = '''\
The following %%d elements have different values: %%s
    %%%ds = %%s
    %%%ds = %%s
    %%%ds = %%s
    ''' % (max_label_length, max_label_length, max_label_length)

    if not np.allclose(np_array1, np_array2):
        conflicts = np.where(np_array1 != np_array2)
        conflict_elements = zip(*conflicts)

        message = message_template % (len(conflict_elements), conflict_elements,
            labels[0], list(np_array1[conflicts]),
            labels[1], list(np_array2[conflicts]),
            labels[2],
            list(np.absolute(np_array2[conflicts] - np_array1[conflicts]
                    ) / np.maximum(np_array2[conflicts],
                        np_array1[conflicts]) * 100))
        raise ValueError, message
