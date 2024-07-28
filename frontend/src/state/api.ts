import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";
// kpis stands for key performance indicators
export const api = createApi({
    baseQuery: fetchBaseQuery({ baseUrl: "http://127.0.0.1:5000" }),
        reducerPath: "main",
        tagTypes: ["EMG_data", "IMU_data"],
        endpoints: (build) => ({
            getEMGData: build.query<void, void>({
                query: () => "/emg_data", 
                providesTags: ["EMG_data"]
            }),
            getIMUData: build.query<void, void>({
                query: () => "/imu_data",
                providesTags: ["IMU_data"]
            })
        }),
})

export const { useGetEMGDataQuery, useGetIMUDataQuery } = api;