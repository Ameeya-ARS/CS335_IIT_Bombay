import json
from typing import Tuple, List
import numpy as np


def generate_uniform(seed: int, num_samples: int) -> None:
    """
    Generate 'num_samples' number of samples from uniform
    distribution and store it in 'uniform.txt'
    """

    # TODO
    np.random.seed(seed)
    result = np.random.uniform(low=0,high=1,size=num_samples)
    file1 = open("uniform.txt",'w')
    for val in result:
        file1.write(f"{val}\n")
    file1.close()
    #np.savetxt("uniform.txt",result)

    # END TODO

    assert len(np.loadtxt("uniform.txt", dtype=float)) == 100
    return None


def inv_transform(file_name: str, distribution: str, **kwargs) -> list:
    """ populate the 'samples' list from the desired distribution """

    samples = []

    # TODO
    file = open(file_name,'r')
    data = file.read()
    org_data_list = data.split("\n")[:-1]
    org_data = np.array(org_data_list,dtype=np.float32)
    file.close()
    if distribution=="exponential":
        lambda_val = kwargs["lambda"]
        result = -np.log(1.0-org_data)/lambda_val
        samples = result.tolist()

    if distribution=="cauchy":
        x0 = kwargs["peak_x"]
        gamma = kwargs["gamma"]
        result = x0 + gamma*np.tan(np.pi*(org_data-0.5))
        samples = result.tolist()

    if distribution=="categorical":
        values = np.array(kwargs["values"])
        cum_probs = np.cumsum(np.array(kwargs["probs"]))
        cum_probs = np.insert(cum_probs,0,0)
        for val in org_data:
            for i in range(cum_probs.size-1):
                if(val>cum_probs[i] and val<=cum_probs[i+1]):
                    samples.append(values[i])
                    break
    # END TODO
    assert len(samples) == 100
    return samples


def find_best_distribution(samples: list) -> Tuple[int, int, int]:
    """
    Given the three distributions of three different types, find the distribution
    which is most likely the data is sampled from for each type
    Return a tupple of three indices corresponding to the best distribution
    of each type as mentioned in the problem statement
    """
    indices = (0,0,0)
    # TODO
    result = []
    data = np.array(samples)
    probs1_1 = np.sum(-np.square(data)/2.0)
    probs1_2 = np.sum(-np.log(0.5) - (2*np.square(data)))
    probs1_3 = np.sum(-np.square(data-1)/2.0)

    a=0
    b=1
    temp=[]
    for val in data:
        if(val<a):
            temp.append(0)
        elif(val>b):
            temp.append(1)
        else:
            temp.append(1/(b-a))
    probs2_1 = np.prod(np.array(temp,dtype=np.float32))

    a=0
    b=2
    for val in data:
        if(val<a):
            temp.append(0)
        elif(val>b):
            temp.append(1)
        else:
            temp.append(1/(b-a))
    probs2_2 = np.prod(np.array(temp,dtype=np.float32))

    a=-1
    b=1
    for val in data:
        if(val<a):
            temp.append(0)
        elif(val>b):
            temp.append(1)
        else:
            temp.append(1/(b-a))
    probs2_3 = np.prod(np.array(temp,dtype=np.float32))

    probs3_1 = np.sum(np.log(0.5) - 0.5*data)
    probs3_2 = np.sum(-data)
    probs3_3 = np.sum(np.log(2) - 2*data)

    if(probs1_1>probs1_2 and probs1_1>probs1_3):
        result.append(0)
    elif(probs1_2>probs1_1 and probs1_2>probs1_3):
        result.append(1)
    elif(probs1_3>probs1_1 and probs1_3>probs1_2):
        result.append(2)
    if(probs2_1>probs2_2 and probs2_1>probs2_3):
        result.append(0)
    elif(probs2_2>probs2_1 and probs2_2>probs2_3):
        result.append(1)
    elif(probs2_3>probs2_1 and probs2_3>probs2_2):
        result.append(2)
    if(probs3_1>probs3_2 and probs3_1>probs3_3):
        result.append(0)
    elif(probs3_2>probs3_1 and probs3_2>probs3_3):
        result.append(1)
    elif(probs3_3>probs3_1 and probs3_3>probs3_2):
        result.append(2)

    indices = tuple(result)
    # END TODO
    assert len(indices) == 3
    assert all([index >= 0 and index <= 2 for index in indices])
    return indices

def marks_confidence_intervals(samples: list, variance: float, epsilons: list) -> Tuple[float, List[float]]:

    sample_mean = 0
    deltas = [0 for e in epsilons] # List of zeros

    # TODO
    samples_array = np.array(samples)
    sample_mean = np.mean(samples_array)
    variance = 5
    index = 0
    for val in epsilons:
        temp = variance/(samples_array.size*val*val)
        deltas[index] = temp
        index = index+1
    # END TODO

    assert len(deltas) == len(epsilons)
    return sample_mean, deltas

if __name__ == "__main__":
    seed = 21734

    # question 1
    generate_uniform(seed, 100)

    # question 2
    for distribution in ["categorical", "exponential", "cauchy"]:
        file_name = "q2_" + distribution + ".json"
        args = json.load(open(file_name, "r"))
        samples = inv_transform(**args)
        with open("q2_output_" + distribution + ".txt", "w") as f:
            for elem in samples:
                f.write(str(elem) + "\n")

    # question 3
    indices = find_best_distribution(np.loadtxt("q3_samples.csv", dtype=float))
    with open("q3_output.txt", "w") as f:
        f.write("\n".join([str(e) for e in indices]))

    # question 4
    q4_samples = np.loadtxt("q4_samples.csv", dtype=float)
    q4_epsilons = np.loadtxt("q4_epsilons.csv", dtype=float)
    variance = 5

    sample_mean, deltas = marks_confidence_intervals(q4_samples, variance, q4_epsilons)

    with open("q4_output.txt", "w") as f:
        f.write("\n".join([str(e) for e in [sample_mean, *deltas]]))
