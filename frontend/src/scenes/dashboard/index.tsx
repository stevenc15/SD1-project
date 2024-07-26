import { Box, useMediaQuery } from '@mui/material'
import Row1 from './Row1';
import Row2 from './Row2';
import Row3 from './Row3';
import FlexBetween from '@/components/FlexBetween';
import DashboardBox from '@/components/DashboardBox';
import Row4 from './Row4';
import Row5 from './Row5';
import Row6 from './Row6';

const gridTemplateLargeScreens = `
    "z z z"

    "a a b"
    "a a b"    
    "a a b"
    "a a c"
    "a a c"
    "a a c"

    "g g g"

    "d d e"
    "d d e"    
    "d d e"
    "d d f"
    "d d f"
    "d d f"
    
    "h h h"

    "i i j"
    "i i j"
    "i i j"
    "i i k"
    "i i k"
    "i i k"
    
    "l l l"
    
    "m m n"
    "m m n"
    "m m n"
    "m m o"
    "m m o"
    "m m o"
    
    "p p p"

    "q q r"
    "q q r"
    "q q r"
    "q q s"
    "q q s"
    "q q s"
    
    "t t t"

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
    const isAboveMediumScreens = useMediaQuery("(min-width: 1200px)")
  return (
    // Setting up the grid of the Dashboard. Check https://grid.malven.co/ and https://developer.mozilla.org/en-US/docs/Web/CSS/grid-template-areas for reference
    
    <Box width="100%" height="100%" display="grid" gap="1.5rem"
        sx={
            isAboveMediumScreens ? {
                // modify the minmax for the size of the graphs
                // fr -> fractional units (split each box evenly)
            gridTemplateColumns: "repeat(3, minmax(370px, 1fr))",
            gridTemplateRows: "repeat(42, minmax(40px, 1fr))",
            gridTemplateAreas: gridTemplateLargeScreens,
        } : {
            gridAutoColumns: "1fr",
            gridAutoRows: "80px",
            gridTemplateAreas: gridTemplateSmallScreens,
        }}>
            <DashboardBox gridArea="z" width="200px">SENSOR 1 </DashboardBox>
            <Row1></Row1>
            <DashboardBox gridArea="g" width="200px">SENSOR 2</DashboardBox>
            <Row2></Row2>
            <DashboardBox gridArea="h" width="200px">SENSOR 3 </DashboardBox>
            <Row3></Row3>
            <DashboardBox gridArea="l" width="200px">SENSOR 4 </DashboardBox>
            <Row4></Row4>
            <DashboardBox gridArea="p" width="200px">SENSOR 5 </DashboardBox>
            <Row5></Row5>
            <DashboardBox gridArea="t" width="200px">SENSOR 6 </DashboardBox>
            <Row6></Row6>
        </Box>
  )
}

export default Dashboard