import { Box, CircularProgress, useMediaQuery } from '@mui/material'
import Row1 from './Row1';
import Row2 from './Row2';
import Row3 from './Row3';
import Row4 from './Row4';
import Row5 from './Row5';
import Row6 from './Row6';
// import { useGetEMGDataQuery, useGetIMUDataQuery } from '@/state/api';
// import { useEffect, useState } from 'react';
// import { useData } from '@/context/DataContext';
import { useGetEMGDataQuery, useGetIMUDataQuery } from '@/state/api';

const gridTemplateLargeScreens = `
    "a a b"
    "a a b"    
    "a a b"
    "a a c"
    "a a c"
    "a a c"
    "d d e"
    "d d e"    
    "d d e"
    "d d f"
    "d d f"
    "d d f"
    "i i j"
    "i i j"
    "i i j"
    "i i k"
    "i i k"
    "i i k"
    "m m n"
    "m m n"
    "m m n"
    "m m o"
    "m m o"
    "m m o"
    "q q r"
    "q q r"
    "q q r"
    "q q s"
    "q q s"
    "q q s"
    "u u v"
    "u u v"
    "u u v"
    "u u w"
    "u u w"
    "u u w"
`;

const gridTemplateSmallScreens = `
    "z"
    "a"
    "a"
    "a"
    "a"
    "a"
    "a"
    "b"
    "b"
    "b"
    "c"
    "c"
    "c"
    "g"
    "d"
    "d"
    "d"
    "d"
    "d"
    "d"
    "e"
    "e"
    "e"
    "f"
    "f"
    "f"
    "h"
    "i"
    "i"
    "i"
    "i"
    "i"
    "i"
    "j"
    "j"
    "j"
    "k"
    "k"
    "k"
    "l"
    "m"
    "m"
    "m"
    "m"
    "m"
    "m"
    "n"
    "n"
    "n"
    "o"
    "o"
    "o"
    "p"
    "q"
    "q"
    "q"
    "q"
    "q"
    "q"
    "r"
    "r"
    "r"
    "s"
    "s"
    "s"
    "t"
    "u"
    "u"
    "u"
    "u"
    "u"
    "u"
    "v"
    "v"
    "v"
    "w"
    "w"
    "w"
`;

const Dashboard = () => {
    
    // const { emgData, imuData, emgLoading, imuLoading, emgError, imuError } = useData();
    const { data: emgData, isError: emgError, isLoading: emgLoading } = useGetEMGDataQuery();
    const {data: imuData, isError: imuError, isLoading: imuLoading} = useGetIMUDataQuery()
    const isAboveMediumScreens = useMediaQuery("(min-width: 1200px)");

    if (emgLoading || imuLoading) {
        return (<CircularProgress />)
    }

    if (emgError || imuError) {
        return (<div>Error loading data. Message from index.tsx</div>)
    }

  return (
    // Setting up the grid of the Dashboard. Check https://grid.malven.co/ and https://developer.mozilla.org/en-US/docs/Web/CSS/grid-template-areas for reference
    
    <Box width="100%" height="100%" display="grid" gap="1.5rem"
        sx={
            isAboveMediumScreens ? {
                // modify the minmax for the size of the graphs
                // fr -> fractional units (split each box evenly)
            gridTemplateColumns: "repeat(3, minmax(370px, 1fr))",
            gridTemplateRows: "repeat(36, minmax(90px, 1fr))",
            gridTemplateAreas: gridTemplateLargeScreens,
        } : {
            gridAutoColumns: "1fr",
            gridAutoRows: "80px",
            gridTemplateAreas: gridTemplateSmallScreens,
        }}>
            <Row1 emg_data={emgData} imu_data={imuData}></Row1>
            <Row2 emg_data={emgData} imu_data={imuData}></Row2>
            <Row3 emg_data={emgData} imu_data={imuData}></Row3>
            <Row4 emg_data={emgData} imu_data={imuData}></Row4>
            <Row5 emg_data={emgData} imu_data={imuData}></Row5>
            <Row6 emg_data={emgData} imu_data={imuData}></Row6>
        </Box>
  )
}

export default Dashboard