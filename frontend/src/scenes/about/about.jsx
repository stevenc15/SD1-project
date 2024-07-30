import { Box, Avatar, IconButton, Divider, Typography } from "@mui/material";
import DashboardBox from "@/components/DashboardBox";
import GitHubIcon from "@mui/icons-material/GitHub";
import LinkedInIcon from "@mui/icons-material/LinkedIn";

function About() {
  const gridTemplateLargeScreens = `
    "a b"
    "c d"
`;
  return (
    <Box>
      <Box
        sx={{
          display: "flex",
          alignItems: "flex-start",
          flexWrap: "wrap",
          gap: 1.5,
          "& > div": { minWidth: "clamp(0px, (150px - 100%) * 999 ,100%)" },
          gridTemplateColumns: "repeat(2, minmax(370px, 1fr))",
          gridTemplateRows: "repeat(2, minmax(90px, 1fr))",
          gridTemplateAreas: gridTemplateLargeScreens,
        }}
      >
        <DashboardBox gridArea="a">
          <Box sx={{ position: "relative", display: "inline-block" }}>
            <Avatar
              variant="rounded"
              imgProps={{
                width: "70",
                height: "70",
                loading: "lazy",
              }}
              alt="Oliver Fritsche"
              src="/public/icon1.jpeg"
              sx={(theme) => ({
                width: 70,
                height: 70,
                borderRadius: 1,
                border: "1px solid",
                borderColor: "grey.100",
                backgroundColor: "primary.50",
              })}
            />
            <Box
              sx={(theme) => ({
                width: 24,
                height: 24,
                display: "flex",
                justifyContent: "center",
                position: "absolute",
                bottom: 0,
                right: 0,
                backgroundColor: "#FFF",
                borderRadius: 40,
                border: "2px solid",
                borderColor: "primary.50",
                boxShadow: "0px 2px 6px rgba(0, 0, 0, 0.1)",
                transform: "translateX(50%)",
                overflow: "hidden",
              })}
            >
              <img
                loading="lazy"
                height="20"
                width="40"
                src="https://flagcdn.com/us.svg"
                alt=""
              />
            </Box>
          </Box>
          <Box sx={{ mt: -0.5, mr: -0.5, ml: "auto" }}>
            <IconButton
              aria-label="Oliver Fritsche LinkedIn profile"
              component="a"
              href="https://github.com/oliverkristianfritsche"
              target="_blank"
              rel="noopener"
            >
              <GitHubIcon fontSize="small" sx={{ color: "grey.500" }} />
            </IconButton>
            <IconButton
              aria-label="Oliver Fritsche LinkedIn profile"
              component="a"
              href="https://www.linkedin.com/in/oliverfritsche/"
              target="_blank"
              rel="noopener"
            >
              <LinkedInIcon fontSize="small" sx={{ color: "grey.500" }} />
            </IconButton>
          </Box>
          <Typography
            variant="body2"
            sx={{ fontWeight: "bold", mt: 2, mb: 0.5 }}
          >
            Oliver Fritsche
          </Typography>
          <Typography variant="body2" sx={{ color: "text.secondary" }}>
            Machine Learning
          </Typography>
          <Divider sx={{ my: 1.5 }} />
          <Typography variant="body2" sx={{ color: "text.tertiary" }}>
            Fulcrum
          </Typography>
        </DashboardBox>
        <DashboardBox gridArea="b">
          <Box sx={{ position: "relative", display: "inline-block" }}>
            <Avatar
              variant="rounded"
              imgProps={{
                width: "70",
                height: "70",
                loading: "lazy",
              }}
              alt="Steven Carmacho"
              src="/public/icon2.jpeg"
              sx={(theme) => ({
                width: 70,
                height: 70,
                borderRadius: 1,
                border: "1px solid",
                borderColor: "grey.100",
                backgroundColor: "primary.50",
              })}
            />
            <Box
              sx={(theme) => ({
                width: 24,
                height: 24,
                display: "flex",
                justifyContent: "center",
                position: "absolute",
                bottom: 0,
                right: 0,
                backgroundColor: "#FFF",
                borderRadius: 40,
                border: "2px solid",
                borderColor: "primary.50",
                boxShadow: "0px 2px 6px rgba(0, 0, 0, 0.1)",
                transform: "translateX(50%)",
                overflow: "hidden",
              })}
            >
              <img
                loading="lazy"
                height="20"
                width="40"
                src="https://flagcdn.com/pr.svg"
                alt=""
              />
            </Box>
          </Box>
          <Box sx={{ mt: -0.5, mr: -0.5, ml: "auto" }}>
            <IconButton
              aria-label="Steven Carmacho GitHub profile"
              component="a"
              href="https://github.com/stevenc15"
              target="_blank"
              rel="noopener"
            >
              <GitHubIcon fontSize="small" sx={{ color: "grey.500" }} />
            </IconButton>
            <IconButton
              aria-label="Steven Carmacho LinkedIn profile"
              component="a"
              href="https://www.linkedin.com/in/steven-camacho-96a818268/"
              target="_blank"
              rel="noopener"
            >
              <LinkedInIcon fontSize="small" sx={{ color: "grey.500" }} />
            </IconButton>
          </Box>
          <Typography
            variant="body2"
            sx={{ fontWeight: "bold", mt: 2, mb: 0.5 }}
          >
            Oliver Fritsche
          </Typography>
          <Typography variant="body2" sx={{ color: "text.secondary" }}>
            Machine Learning
          </Typography>
          <Divider sx={{ my: 1.5 }} />
          <Typography variant="body2" sx={{ color: "text.tertiary" }}>
            Come in
          </Typography>
        </DashboardBox>
        <DashboardBox gridArea="c">
          <Box sx={{ position: "relative", display: "inline-block" }}>
            <Avatar
              variant="rounded"
              imgProps={{
                width: "70",
                height: "70",
                loading: "lazy",
              }}
              alt="Carlos Arciniegas"
              src="/public/icon3.jpeg"
              sx={(theme) => ({
                width: 70,
                height: 70,
                borderRadius: 1,
                border: "1px solid",
                borderColor: "grey.100",
                backgroundColor: "primary.50",
              })}
            />
            <Box
              sx={(theme) => ({
                width: 24,
                height: 24,
                display: "flex",
                justifyContent: "center",
                position: "absolute",
                bottom: 0,
                right: 0,
                backgroundColor: "#FFF",
                borderRadius: 40,
                border: "2px solid",
                borderColor: "primary.50",
                boxShadow: "0px 2px 6px rgba(0, 0, 0, 0.1)",
                transform: "translateX(50%)",
                overflow: "hidden",
              })}
            >
              <img
                loading="lazy"
                height="20"
                width="40"
                src="https://flagcdn.com/co.svg"
                alt=""
              />
            </Box>
          </Box>
          <Box sx={{ mt: -0.5, mr: -0.5, ml: "auto" }}>
            <IconButton
              aria-label="Carlos Arciniegas GitHub profile"
              component="a"
              href="https://github.com/cdam123"
              target="_blank"
              rel="noopener"
            >
              <GitHubIcon fontSize="small" sx={{ color: "grey.500" }} />
            </IconButton>
            <IconButton
              aria-label="Carlos Arciniegas LinkedIn profile"
              component="a"
              href="https://www.linkedin.com/in/carlos-arciniegads/"
              target="_blank"
              rel="noopener"
            >
              <LinkedInIcon fontSize="small" sx={{ color: "grey.500" }} />
            </IconButton>
          </Box>
          <Typography
            variant="body2"
            sx={{ fontWeight: "bold", mt: 2, mb: 0.5 }}
          >
            Carlos Arciniegas
          </Typography>
          <Typography variant="body2" sx={{ color: "text.secondary" }}>
            Frontend, Backend
          </Typography>
          <Divider sx={{ my: 1.5 }} />
          <Typography variant="body2" sx={{ color: "text.tertiary" }}>
            Gang shit
          </Typography>
        </DashboardBox>
        <DashboardBox gridArea="d">
          <Box sx={{ position: "relative", display: "inline-block" }}>
            <Avatar
              variant="rounded"
              imgProps={{
                width: "70",
                height: "70",
                loading: "lazy",
              }}
              alt="Tyler Halfpenny"
              src="/public/icon4.jpeg"
              sx={(theme) => ({
                width: 70,
                height: 70,
                borderRadius: 1,
                border: "1px solid",
                borderColor: "grey.100",
                backgroundColor: "primary.50",
              })}
            />
            <Box
              sx={(theme) => ({
                width: 24,
                height: 24,
                display: "flex",
                justifyContent: "center",
                position: "absolute",
                bottom: 0,
                right: 0,
                backgroundColor: "#FFF",
                borderRadius: 40,
                border: "2px solid",
                borderColor: "primary.50",
                boxShadow: "0px 2px 6px rgba(0, 0, 0, 0.1)",
                transform: "translateX(50%)",
                overflow: "hidden",
              })}
            >
              <img
                loading="lazy"
                height="20"
                width="40"
                src="https://flagcdn.com/us.svg"
                alt=""
              />
            </Box>
          </Box>
          <Box sx={{ mt: -0.5, mr: -0.5, ml: "auto" }}>
            <IconButton      
              aria-label="Tyler Halfpenny GitHub profile"
              component="a"
              href="https://github.com/tchalfpenny"
              target="_blank"
              rel="noopener"
            >
              <GitHubIcon fontSize="small" sx={{ color: "grey.500" }} />
            </IconButton>
            <IconButton
              aria-label="Tyler Halfpenny LinkedIn profile"
              component="a"
              href="https://www.linkedin.com/in/tylerhalfpenny/"
              target="_blank"
              rel="noopener"
            >
              <LinkedInIcon fontSize="small" sx={{ color: "grey.500" }} />
            </IconButton>
          </Box>
          <Typography
            variant="body2"
            sx={{ fontWeight: "bold", mt: 2, mb: 0.5 }}
          >
            Tyler Halfpenny
          </Typography>
          <Typography variant="body2" sx={{ color: "text.secondary" }}>
            Frontend
          </Typography>
          <Divider sx={{ my: 1.5 }} />
          <Typography variant="body2" sx={{ color: "text.tertiary" }}>
            Penjamin
          </Typography>
        </DashboardBox>
      </Box>
    </Box>
  );
}

export default About;
