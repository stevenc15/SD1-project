
#IMPORTS
from sklearn.metrics import mean_squared_error #used in evaluating predictions
import numpy as np
def prediction_test(yhat,test_y_up):

    test_o=test_y_up
    yhat=yhat

    y_1=yhat[:,0]
    y_2=yhat[:,1]
    y_3=yhat[:,2]


    y_test_1=test_o[:,0]
    y_test_2=test_o[:,1]
    y_test_3=test_o[:,2]


    ###calculate RMSE

    rmse_1 =np.sqrt(mean_squared_error(y_test_1,y_1))
    rmse_2 =np.sqrt(mean_squared_error(y_test_2,y_2))
    rmse_3 =np.sqrt(mean_squared_error(y_test_3,y_3))

    p_1=np.corrcoef(y_1, y_test_1)[0, 1]
    p_2=np.corrcoef(y_2, y_test_2)[0, 1]
    p_3=np.corrcoef(y_3, y_test_3)[0, 1]

    ### Getiing single RMSE and PCC value for a joint
    p=((p_1+p_2+p_3)/3)

    rmse=((rmse_1+rmse_2+rmse_3)/3)



    return rmse,p