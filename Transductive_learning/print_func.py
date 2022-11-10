from collections  import OrderedDict


def func_print(train_catboost, train_mlp, train_graph_v1, train_graph_v2, outfile):
    
    output=OrderedDict()
    for key,val in train_catboost.items():
        output[key]=[train_catboost[key],train_mlp[key],train_graph_v1[key],train_graph_v2[key]]

    dict_key=list(output.keys())

    with open(outfile,'a') as f:
        f.write('{:<30}{:<20}{:<20}{:<20}{:<20}\n'.format("--"*20,"--"*10,"--"*10,"--"*10,"--"*10))
        print('{:<30}{:<20}{:<20}{:<35}{:<25}'.format(dict_key[0], output[dict_key[0]][0], output[dict_key[0]][1], output[dict_key[0]][2], output[dict_key[0]][3]))
        f.write('{:<30}{:<20}{:<20}{:<35}{:<25}\n'.format(dict_key[0], output[dict_key[0]][0], output[dict_key[0]][1], output[dict_key[0]][2], output[dict_key[0]][3]))
        print('{:<30}{:<20}{:<20}{:<35}{:<25}'.format(" "," "," "," "," "))
        f.write('{:<30}{:<20}{:<20}{:<35}{:<25}\n'.format(" "," "," "," "," "))
        for i in range(1,len(dict_key)):
            if i==1:
                print('{:<30}{:<20}{:<20}{:<35}{:<25}'.format(dict_key[i], output[dict_key[i]][0], output[dict_key[i]][1], 
                                                        output[dict_key[i]][2],output[dict_key[i]][3]))
                f.write('{:<30}{:<20}{:<20}{:<35}{:<25}\n'.format(dict_key[i], output[dict_key[i]][0], output[dict_key[i]][1], 
                                                            output[dict_key[i]][2],output[dict_key[i]][3]))
            elif i<=4:
                print('{:<30}{:<20,}{:<20,}{:<35,}{:<25,}'.format(dict_key[i], output[dict_key[i]][0], output[dict_key[i]][1], 
                                                           output[dict_key[i]][2],output[dict_key[i]][3]))
                f.write('{:<30}{:<20,}{:<20,}{:<35,}{:<25,}\n'.format(dict_key[i], output[dict_key[i]][0], output[dict_key[i]][1], 
                                                               output[dict_key[i]][2],output[dict_key[i]][3]))
            else:
                print('{:<30}{:<20,.2%}{:<20,.2%}{:<35,.2%}{:<25,.2%}'.format(dict_key[i], output[dict_key[i]][0], output[dict_key[i]][1], 
                                                                    output[dict_key[i]][2],output[dict_key[i]][3]))
                f.write('{:<30}{:<20,.2%}{:<20,.2%}{:<35,.2%}{:<25,.2%}\n'.format(dict_key[i], output[dict_key[i]][0], output[dict_key[i]][1], 
                                                                        output[dict_key[i]][2],output[dict_key[i]][3]))
        f.write('{:<30}{:<20}{:<20}{:<20}{:<20}\n'.format("--"*20,"--"*10,"--"*10,"--"*10,"--"*10))
        f.write('{:<30}{:<20}{:<20}{:<20}{:<20}\n'.format(" "," "," "," "," "))
        
