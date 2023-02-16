# Uni-AD

### Share or Not Share? Towards the Practicability of Deep Models for Unsupervised Anomaly Detection in Modern Online Systems (ISSRE'22))

**Best Paper Award at ISSRE 2022 Research Track!**

Anomaly detection is crucial in the management of modern online systems. Due to the complexity of patterns in the monitoring data and the lack of labelled data with anomalies, recent studies mainly adopt deep unsupervised models to address this problem. Notably, even though these models have achieved a great success on experimental datasets, there are still several challenges for them to be successfully applied in a real-world modern online system. Such challenges stem from some significant properties of modern online systems, e.g., large scale, diversity and dynamics. This study investigates how these properties affect the adoption of deep anomaly detectors in modern online systems. Furthermore, we claim that model sharing is an effective way to overcome these challenges. To support this claim, we systematically study the feasibility and necessity of model sharing for unsupervised anomaly detection. In addition, we further propose a novel model, Uni-AD, which works well for model sharing. Based upon Transformer encoder layers and Base layers, Uni-AD can effectively model diverse patterns for different monitored entities and further perform anomaly detection accurately. Besides, it can accept variable-length inputs, which is a required property for a model that needs to be shared. Extensive experiments on two real-world large-scale datasets demonstrate the effectiveness and practicality of Uni-AD. 

The paper can be downloaded from [issre2022_uniad.pdf](./issre2022_uniad.pdf).

## Start

#### Clone the repository

```
git clone https://github.com/IntelligentDDS/Uni-AD.git
```

#### Get data

You can get the public datasets from:

* CTF_data: <https://github.com/NetManAIOps/CTF_data>
* SMD: <https://github.com/NetManAIOps/OmniAnomaly>
* JS_D2+D3: <https://github.com/NetManAIOps/JumpStarter>

Here, since we focus on anomaly detection in modern **large-scale** online systems, we may prioritize CTF_data. 

#### Install dependencies (with python 3.7.6) 

```
pip install -r requirements.txt
```

#### Running an Example

An example using the CTF_data is provided in the notebook `example.ipynb`.

## Cite

Please cite our ISSRE'22 paper if you find this work is helpful.
