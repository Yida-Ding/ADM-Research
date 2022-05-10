from VNSSolver import 

if __name__ == '__main__':
    
#    for i in range(15,35,5):
#        for typ in ['m','p']:
#            for seed in range(5):
#            
#                config1 = {"DATASET": "ACF%d"%i,
#                          "SCENARIO": "ACF%d-SC%s"%(i,typ),
#                          "SEED": seed,
#                          "EPISODE": 20000,
#                          "TRAJLEN": 10,
#                          "FC1DIMS": 256,
#                          "FC2DIMS": 256,
#                          "ALPHA": 0.00001, # smaller, more likely to jump
#                          "GAMMA": 0.9,
#                          "GAELAMBDA": 0.95,
#                          "POLICYCLIP": 0.8,    # Very Important! large clip leads to more exploration and better result
#                          "BATCHSIZE": 20,
#                          "NEPOCH": 3,
#                          "SAVERESULT": True,
#                          "SAVEPOLICY": False
#                          }
