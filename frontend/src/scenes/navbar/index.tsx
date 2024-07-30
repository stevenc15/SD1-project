import { useState } from "react";
import { Link } from "react-router-dom";
import { Box, Typography, useTheme, IconButton } from "@mui/material";
import FlexBetween from "@/components/FlexBetween";
import DataThresholdingIcon from "@mui/icons-material/DataThresholding";
import GitHubIcon from "@mui/icons-material/GitHub";

type Props = {};

const Navbar = (props: Props) => {
  const { palette } = useTheme();
  const [selected, setSelected] = useState("dashboard");
  return (
    <FlexBetween mb="1.2rem" p="2.5 rem 0rem" color={palette.grey[300]}>
      {/* LEFT SIDE */}
      <FlexBetween gap="0.75rem">
        <DataThresholdingIcon sx={{ fontSize: "58px" }} />
        <Typography variant="h4" fontSize="20px">
          EMG IMU Dashboard
        </Typography>
      </FlexBetween>
      {/* RIGHT SIDE */}
      <FlexBetween gap="2rem">
        <Box>
          <IconButton
            aria-label="GitHub Repository"
            component="a"
            href="https://github.com/stevenc15/SD1-project"
            target="_blank"
            rel="noopener"
          >
            <GitHubIcon fontSize="small" sx={{ color: "grey.500" }} />
          </IconButton>
        </Box>
        <Box>
          <Link
            to="/"
            onClick={() => setSelected("dashboard")}
            style={{
              color: selected === "dashboard" ? "inherit" : palette.grey[700],
              textDecoration: "inherit",
              fontSize: "18px",
            }}
          >
            Dashboard
          </Link>
        </Box>
        <Box>
          <Link
            to="/predictions"
            onClick={() => setSelected("predictions")}
            style={{
              color: selected === "predictions" ? "inherit" : palette.grey[700],
              textDecoration: "inherit",
              fontSize: "18px",
            }}
          >
            Predictions
          </Link>
        </Box>
        <Box>
          <Link
            to="/collection"
            onClick={() => setSelected("collection")}
            style={{
              color: selected === "collection" ? "inherit" : palette.grey[700],
              textDecoration: "inherit",
              fontSize: "18px",
            }}
          >
            Collection
          </Link>
        </Box>
        <Box>
          <Link
            to="/about"
            onClick={() => setSelected("about")}
            style={{
              color: selected === "about" ? "inherit" : palette.grey[700],
              textDecoration: "inherit",
              fontSize: "18px",
            }}
          >
            About Us
          </Link>
        </Box>
      </FlexBetween>
    </FlexBetween>
  );
};

export default Navbar;
