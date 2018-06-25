'''
Created on Jun 21, 2018

@author: Moritz
'''

import numpy as np
import matplotlib.pyplot as plt

EPS = 0.00000001


def integral(m, b, x):
    return 0.5 * m * (x ** 2) + b * x


def inverse_integral(m, b, y):
    p = b/ (0.5 * m)
    q = y/(0.5 * m)
    
    #Our scenario is easy we exactly know what value we want
    if m >= 0:
        #Left value is important
        return - (p/2) + np.sqrt((p/2)**2 + q)
    else:
        #Right value is important
        return - (p/2) - np.sqrt((p/2)**2 + q)


def cumulative(cumulative_stats, x):
    
    possible_bin_ids = np.where((cumulative_stats[:,2] <= x) & (x <= cumulative_stats[:,3]))
    bin_id = possible_bin_ids[0][0] 
    stats = cumulative_stats[bin_id]
    
    return integral(stats[0], stats[1], x) - stats[4] + np.sum(cumulative_stats[:bin_id,6])
    

def inverse_cumulative(cumulative_stats, y):
    
    bin_probs = cumulative_stats[:,6]
    cumulative_probs = np.cumsum(bin_probs)
    
    # +EPS to avoid incorrect behavior caused by floating point errors
    cumulative_probs[-1] += EPS
    
    bin_id = np.where(( y <= cumulative_probs))[0][0]
    stats = cumulative_stats[bin_id]
    
    lower_cumulative_prob = 0 if bin_id == 0 else cumulative_probs[bin_id-1]
    y_perc = (y - lower_cumulative_prob) / bin_probs[bin_id]
    y_val = (stats[5] - stats[4]) * y_perc + stats[4]

    return inverse_integral(stats[0], stats[1], y_val)
    

    
def compute_cumulative_stats(x_range, y_range):
    
    #Compute representation of linear functions and cumulative probability
    cumulative_stats = []
    for i in range(len(x_range) - 1):
        y0 = y_range[i]
        y1 = y_range[i + 1]
        x0 = x_range[i]
        x1 = x_range[i + 1]

        m = (y0 - y1) / (x0 - x1)
        b = (x0 * y1 - x1 * y0) / (x0 - x1)
        
        lowest_y = integral(m,b, x0)
        highest_y = integral(m,b, x1)
        prob = highest_y - lowest_y
        
        cumulative_stats.append([m, b, x0, x1, lowest_y, highest_y, prob])
        
    return np.array(cumulative_stats, dtype=np.float64)



if __name__ == '__main__':
    
    #Create distribution
    y_range = [0., 10, 100, 30, 10, 200, 0.]
    x_range = [0.,  2,  4.,  6,  8,  10, 12]
    x_range, y_range = np.array(x_range), np.array(y_range)
    auc = np.trapz(y_range, x_range)
    y_range = y_range / auc
    
    cumulative_stats = compute_cumulative_stats(x_range, y_range)
    
    
    
    #Plot distribution
    plt.title("Actual distribution")
    plt.plot(x_range, y_range)
    plt.show()
    
    
    
    #Plot cumulative distribution
    x_domain = np.linspace(np.min(x_range), np.max(x_range), 100)
    y_domain = np.zeros(len(x_domain))
    for i, x_val in enumerate(x_domain):
        y_domain[i] = cumulative(cumulative_stats, x_val)
    
    plt.title("Cumulative distribution")
    plt.plot(x_domain, y_domain)
    plt.show()
    
    
    
    #Plot inverse cumulative distribution
    x_domain = np.linspace(0, 1, 100)
    y_domain = np.zeros(len(x_domain))
    for i, x_val in enumerate(x_domain):
        y_domain[i] = inverse_cumulative(cumulative_stats, x_val)

    plt.title("Inverse cumulative distribution")
    plt.plot(x_domain, y_domain)
    plt.show()
    