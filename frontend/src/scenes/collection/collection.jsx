import { Box } from "@mui/material";
import DashboardBox from "@/components/DashboardBox";
import demo from "/public/IMG_2850.mov";
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';

function Collection() {
  const gridTemplateLargeScreens = `
    "a b"
    "c d"
    "e e"
`;
  return (
    <>
      <Box
        width="100%"
        height="100%"
        display="grid"
        gap="1.5rem"
        sx={{
          // modify the minmax for the size of the graphs
          // fr -> fractional units (split each box evenly)
          gridTemplateColumns: "repeat(2, minmax(370px, 1fr))",
          gridTemplateRows: "repeat(2, minmax(450px, 1fr))",
          gridTemplateAreas: gridTemplateLargeScreens,
        }}
      >
        <DashboardBox gridArea="a" justifyContent={"space-evenly"}>
            <Card>
                <Box>
                    <CardContent>
                    <Box>
                <p>Arm Swing Trials</p>
                <p>4 Speeds: Slow, Normal, Fast, Very Fast</p>
                <p>30 seconds per trial.</p>
                <p>Full natural range of flexion-extension and abduction-adduction</p>
            </Box>
                    </CardContent>
                </Box>
                <Box>
                    <CardMedia
                        component="video"
                        src={demo}
                        width="300"
                        height="400"
                        autoPlay
                        loop
                    />
                </Box>
            </Card>
            {/* <Box>
                <p>Arm Swing Trials</p>
                <p>4 Speeds: Slow, Normal, Fast, Very Fast</p>
                <p>30 seconds per trial.</p>
                <p>Full natural range of flexion-extension and abduction-adduction</p>
            </Box>
            <video width="300" height="400" loop={true} controls>
                <source src={demo} type="video/mp4" />
            </video> */}
        </DashboardBox>
        <DashboardBox gridArea="b">
            area 2
        </DashboardBox>
        <DashboardBox gridArea="c">
            area 3
        </DashboardBox>
        <DashboardBox gridArea="d">
            area 4
        </DashboardBox>
        <DashboardBox gridArea="e">
            area 5
        </DashboardBox>
      </Box>
    </>
  );
}

export default Collection;
