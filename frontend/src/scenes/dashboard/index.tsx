import { Box, useMediaQuery } from '@mui/material'
import Row1 from './Row1';
import Row2 from './Row2';
import Row3 from './Row3';
import FlexBetween from '@/components/FlexBetween';

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
    "a"
    "a"
    "a"
    "a"
    "b"
    "b"
    "b"
    "b"
    "c"
    "c"
    "c"
    "d"
    "d"
    "d"
    "e"
    "e"
    "f"
    "f"
    "f"
    "g"
    "g"
    "g"
    "h"
    "h"
    "h"
    "h"
    "i"
    "i"
    "j"
    "j"
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
            <Box backgroundColor="#fff" gridArea="z" width="200px">SENSOR 1</Box>
            <Box backgroundColor="#fff" gridArea="a"></Box>
            <Box backgroundColor="#fff" gridArea="b"></Box>
            <Box backgroundColor="#fff" gridArea="c"></Box>
            <Box backgroundColor="#fff" gridArea="d"></Box>
            <Box backgroundColor="#fff" gridArea="e"></Box>
            <Box backgroundColor="#fff" gridArea="f"></Box>
            <Box backgroundColor="#fff" gridArea="g" width="200px">SENSOR 2</Box>
            <Box backgroundColor="#fff" gridArea="h" width="200px">SENSOR 3 </Box>
            <Box backgroundColor="#fff" gridArea="i"> </Box>
            <Box backgroundColor="#fff" gridArea="j"> </Box>
            <Box backgroundColor="#fff" gridArea="k"> </Box>
            <Box backgroundColor="#fff" gridArea="l" width="200px">SENSOR 4 </Box>
            <Box backgroundColor="#fff" gridArea="m"> </Box>
            <Box backgroundColor="#fff" gridArea="n"> </Box>
            <Box backgroundColor="#fff" gridArea="o"> </Box>
            <Box backgroundColor="#fff" gridArea="p" width="200px">SENSOR 5 </Box>
            <Box backgroundColor="#fff" gridArea="q"> </Box>
            <Box backgroundColor="#fff" gridArea="r"> </Box>
            <Box backgroundColor="#fff" gridArea="s"> </Box>
            <Box backgroundColor="#fff" gridArea="t" width="200px">SENSOR 6 </Box>
            <Box backgroundColor="#fff" gridArea="u"> </Box>
            <Box backgroundColor="#fff" gridArea="v"> </Box>
            <Box backgroundColor="#fff" gridArea="w"> </Box>

            {/* <Box backgroundColor="#fff" gridArea="h"></Box> }
            <Box backgroundColor="#fff" gridArea="i"></Box>
            <Box backgroundColor="#fff" gridArea="j"></Box> }
            {/* <Row1/> */}
            {/* <Row2/>
            <Row3/> */}
        </Box>
  )
}

export default Dashboard