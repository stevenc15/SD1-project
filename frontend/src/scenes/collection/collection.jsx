import { Box } from "@mui/material";
import DashboardBox from "@/components/DashboardBox";
import demo from "/public/IMG_2850.mov";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import CardMedia from "@mui/material/CardMedia";
import { useTheme } from "@mui/material";

function Collection() {
  const { palette } = useTheme();
  const gridTemplateLargeScreens = `
    "a b"
    "c d"
    "e f"
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
          // gridTemplateColumns: "repeat(2, minmax(200px, 1fr))",
          // gridTemplateRows: "repeat(3, minmax(200px, 1fr))",
          gridTemplateAreas: gridTemplateLargeScreens,
        }}
      >
        <DashboardBox gridArea="a" justifyContent={"space-evenly"}>
          <Card sx={{ display: "flex", backgroundColor: "sky.200" }}>
            <Box sx={{ display: "flex", flexDirection: "column" }}>
              <CardContent sx={{ flex: "1 0 auto" }}>
                <b>Arm Swing Trials</b>
                <p>Four trials, each at a different speed.</p>
                <p>Speeds include slow, normal, fast, and very fast</p>
                <p>30 seconds per trial.</p>
                <p>
                  Angles of motion consist of the full natural range of
                  flexion-extension and abduction-adduction.
                </p>
                <p>
                  Path of motion consists of free, natural swings in the
                  saggital and frontal planes.
                </p>
              </CardContent>
            </Box>
            <Box>
              <CardMedia
                component="video"
                src={demo}
                width="300"
                height="100%"
                autoPlay
                loop
              />
            </Box>
          </Card>
        </DashboardBox>
        <DashboardBox gridArea="b" justifyContent={"space-evenly"}>
          <Card sx={{ display: "flex", backgroundColor: "sky.200" }}>
            <Box sx={{ display: "flex", flexDirection: "column" }}>
              <CardContent sx={{ flex: "1 0 auto" }}>
                <b>Overhead Reach Trials</b>
                <p>Three trials, each up to a different angle.</p>
                <p>Angles include 90, 180, and maximum participant reach.</p>
                <p>30 seconds per trial.</p>
                <p>Angles of motion range from neutral (0) to maximum elevation.</p>
                <p>Path of motion consists of neutral to directly overhead.</p>
              </CardContent>
            </Box>
            <Box>
              <CardMedia
                component="video"
                src={demo}
                width="300"
                height="100%"
                autoPlay
                loop
              />
            </Box>
          </Card>
        </DashboardBox>
        <DashboardBox gridArea="c" justifyContent={"space-evenly"}>
          <Card sx={{ display: "flex", backgroundColor: "sky.200" }}>
            <Box sx={{ display: "flex", flexDirection: "column" }}>
              <CardContent sx={{ flex: "1 0 auto" }}>
                  <b>Elbow Flexion and Extension</b>
                  <p>Three trials, each at a different speed.</p>
                  <p>Speeds include slow, normal, and fast.</p>
                  <p>30 seconds per trial.</p>
                  <p>
                    Angles of motion range from 0 (full extension) to 
                    150 (full flexion).
                  </p>
                  <p>
                    Path of motion consists of smooth controlled movement along the
                    elbow's hinge path.
                  </p>
              </CardContent>
            </Box>
            <Box>
              <CardMedia
                component="video"
                src={demo}
                width="300"
                height="100%"
                autoPlay
                loop
              />
            </Box>
          </Card>
        </DashboardBox>
        <DashboardBox gridArea="d" justifyContent={"space-evenly"}>
          <Card sx={{ display: "flex", backgroundColor: "sky.200" }}>
            <Box sx={{ display: "flex", flexDirection: "column" }}>
              <CardContent sx={{ flex: "1 0 auto" }}>
                <b>Elbow Rotation Trials</b>
                <p>Three trials, each at a different speed.</p>
                <p>Angles include 90, 180, and maximum participant reach.</p>
                <p>30 seconds per trial.</p>
                <p>
                  Angles of motion range from neutral (0) to 90, internal and
                  external.
                </p>
                <p>
                  Path of motion consists of controlled rotation at the
                  shoulder.
                </p>
              </CardContent>
            </Box>
            <Box>
              <CardMedia
                component="video"
                src={demo}
                width="300"
                height="100%"
                autoPlay
                loop
              />
            </Box>
          </Card>
        </DashboardBox>
        <DashboardBox gridArea="e" justifyContent={"space-evenly"}>
          <Card sx={{ display: "flex", backgroundColor: "sky.200" }}>
            <Box sx={{ display: "flex", flexDirection: "column" }}>
              <CardContent sx={{ flex: "1 0 auto" }}>
                <b>Cross Body Reach Trials</b>
                <p>Three trials, each at a different speed.</p>
                <p>Angles include 90, 180, and maximum participant reach.</p>
                <p>30 seconds per trial.</p>
                <p>
                  Angles of motion range from neutral (0) to 90, internal and external.
                </p>
                <p>
                  Path of motion consists of a diagonal path across the
                </p>
              </CardContent>
            </Box>
            <Box>
              <CardMedia
                component="video"
                src={demo}
                width="300"
                height="100%"
                autoPlay
                loop
              />
            </Box>
          </Card>
        </DashboardBox>
        <DashboardBox gridArea="f" justifyContent={"space-evenly"}>
          <Card sx={{ display: "flex", backgroundColor: "sky.200" }}>
            <Box sx={{ display: "flex", flexDirection: "column" }}>
              <CardContent sx={{ flex: "1 0 auto" }}>
                <b>Elbow Rotation Trials</b>
                <p>Three trials, each at a different speed.</p>
                <p>Angles include 90, 180, and maximum participant reach.</p>
                <p>30 seconds per trial.</p>
                <p>
                  Angles of motion range from neutral (0) to 90, internal and
                  external.
                </p>
                <p>
                  Path of motion consists of controlled rotation at the
                  shoulder.
                </p>
              </CardContent>
            </Box>
            <Box>
              <CardMedia
                component="video"
                src={demo}
                width="300"
                height="100%"
                autoPlay
                loop
              />
            </Box>
          </Card>
        </DashboardBox>
      </Box>
    </>
  );
}

export default Collection;
