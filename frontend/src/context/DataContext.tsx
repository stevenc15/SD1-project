import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { useGetEMGDataQuery, useGetIMUDataQuery } from "@/state/api";

interface DataContextProps {
    emgData: any;
    imuData: any;
    emgLoading: boolean;
    imuLoading: boolean;
    emgError: boolean;
    imuError: boolean;
}

const DataContext = createContext<DataContextProps | undefined>(undefined);

interface DataProviderProps {
    children: ReactNode;
}

export const DataProvider: React.FC<DataProviderProps> = ({ children }) => {
    const { data: emgData, isError: emgError, isLoading: emgLoading } = useGetEMGDataQuery();
    const { data: imuData, isError: imuError, isLoading: imuLoading } = useGetIMUDataQuery();
    
    const [emgDataState, setEmgDataState] = useState<any>(null);
    const [imuDataState, setImuDataState] = useState<any>(null);

    useEffect(() => {
        if (emgData && !emgError && !emgLoading) {
            setEmgDataState(emgData);
        }
    }, [emgData, emgError, emgLoading]);

    useEffect(() => {
        if (imuData && !imuError && !imuLoading) {
            setImuDataState(imuData);
        }
    }, [imuData, imuError, imuLoading]);

    return (
        <DataContext.Provider value={{ emgData: emgDataState, imuData: imuDataState, emgLoading, imuLoading, emgError, imuError }}>
            {children}
        </DataContext.Provider>
    );
}

export const useData = () => {
    const context = useContext(DataContext);
    if (context === undefined) {
        throw new Error("useData must be used within a DataProvider");
    }
    return context;
}
