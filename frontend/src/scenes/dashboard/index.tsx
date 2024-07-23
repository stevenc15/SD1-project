import { Box, useMediaQuery } from '@mui/material'
import Row1 from './Row1';
import Row2 from './Row2';
import Row3 from './Row3';

const gridTemplateLargeScreens = `
    "a a a"
    "b c c"    
    "b c c"
    "d d d"
    "d d d"
    "d d d"
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
            gridTemplateColumns: "repeat(3, minmax(370px, 1fr))",
            gridTemplateRows: "repeat(1, minmax(40px, 1fr))",
            gridTemplateAreas: gridTemplateLargeScreens,
        } : {
            gridAutoColumns: "1fr",
            gridAutoRows: "80px",
            gridTemplateAreas: gridTemplateSmallScreens,
        }}>
            <Row1/>
            <Row2/>
            <Row3/>
        </Box>
  )
}

export default Dashboard