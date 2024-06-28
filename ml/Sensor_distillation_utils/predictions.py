
#IMPORTS
from sklearn.metrics import mean_squared_error #used in evaluating predictions
import numpy as np
def prediction_test(yhat,test_y_up):

    test_o=test_y_up
    yhat=yhat

    y_1=yhat[:,0]
    y_2=yhat[:,1]
    y_3=yhat[:,2]
    y_4=yhat[:,3]
    y_5=yhat[:,4]
    y_6=yhat[:,5]


    y_test_1=test_o[:,0]
    y_test_2=test_o[:,1]
    y_test_3=test_o[:,2]
    y_test_4=test_o[:,3]
    y_test_5=test_o[:,4]
    y_test_6=test_o[:,5]


    ###calculate RMSE

    rmse_1 =np.sqrt(mean_squared_error(y_test_1,y_1))
    rmse_2 =np.sqrt(mean_squared_error(y_test_2,y_2))
    rmse_3 =np.sqrt(mean_squared_error(y_test_3,y_3))
    rmse_4 =np.sqrt(mean_squared_error(y_test_4,y_4))
    rmse_5 =np.sqrt(mean_squared_error(y_test_5,y_5))
    rmse_6 =np.sqrt(mean_squared_error(y_test_6,y_6))


    p_1=np.corrcoef(y_1, y_test_1)[0, 1]
    p_2=np.corrcoef(y_2, y_test_2)[0, 1]
    p_3=np.corrcoef(y_3, y_test_3)[0, 1]
    p_4=np.corrcoef(y_4, y_test_4)[0, 1]
    p_5=np.corrcoef(y_5, y_test_5)[0, 1]
    p_6=np.corrcoef(y_6, y_test_6)[0, 1]

    ### Getiing single RMSE and PCC value for a joint
    p=np.array([(p_1+p_4)/2,(p_2+p_5)/2,(p_3+p_6)/2])

    rmse=np.array([(rmse_1+rmse_4)/2,(rmse_2+rmse_5)/2,(rmse_3+rmse_6)/2])



    return rmse,p