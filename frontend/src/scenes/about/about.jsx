import { Box, Avatar, IconButton, Divider, Typography, useTheme } from "@mui/material";
import DashboardBox from "@/components/DashboardBox";
import GitHubIcon from "@mui/icons-material/GitHub";
import LinkedInIcon from "@mui/icons-material/LinkedIn";

function About() {
  const { palette } = useTheme();
  const gridTemplateLargeScreens = `
    "a b"
    "c d"
`;
  return (
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexWrap: "wrap",
          gap: 1.5,
          width: "100%",
          height: "100%",
          gridTemplateAreas: gridTemplateLargeScreens,
        }}
      >
        <DashboardBox gridArea="a" sx={{backgroundColor: "sky.200"}}>
          <Box sx={{ position: "relative", display: "inline-block" }}>
            <Avatar
              variant="rounded"
              imgProps={{
                width: "150",
                height: "150",
                loading: "lazy",
              }}
              alt="Oliver Fritsche"
              src="/public/icon1.jpeg"
              sx={(theme) => ({
                width: 150,
                height: 150,
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
                transform: "translateX(-100%), translateY(100%)",
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
          <Box sx={{ mt: -0.5, ml: "auto", backgroundColor:"sky.200"}}>
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
            sx={{ fontWeight: "bold", ml: 1 , mt: 2, mb: 0.5}}
          >
              Oliver Fritsche
          </Typography>
          <Typography variant="body2" sx={{ ml:1, color: "text.secondary" }}>
            Machine Learning
          </Typography>
        </DashboardBox>
        <DashboardBox gridArea="b" sx={{backgroundColor: "sky.200"}}>
          <Box sx={{ position: "relative", display: "inline-block" }}>
            <Avatar
              variant="rounded"
              imgProps={{
                width: "150",
                height: "150",
                loading: "lazy",
              }}
              alt="Steven Carmacho"
              src="/public/icon2.jpeg"
              sx={(theme) => ({
                width: 150,
                height: 150,
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
                transform: "translateX(-100%), translateY(100%)",
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
            sx={{ fontWeight: "bold", ml:1, mt: 2, mb: 0.5 }}
          >
              Steven Carmacho
          </Typography>
          <Typography variant="body2" sx={{ ml:1, color: "text.secondary" }}>
              Machine Learning
          </Typography>
        </DashboardBox>
        <DashboardBox gridArea="c" sx={{backgroundColor: "sky.200"}}>
          <Box sx={{ position: "relative", display: "inline-block" }}>
            <Avatar
              variant="rounded"
              imgProps={{
                width: "150",
                height: "150",
                loading: "lazy",
              }}
              alt="Carlos Arciniegas"
              src="/public/icon3.jpeg"
              sx={(theme) => ({
                width: 150,
                height: 150,
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
                transform: "translateX(-100%), translateY(100%)",
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
            sx={{ fontWeight: "bold", ml:1, mt: 2, mb: 0.5 }}
          >
            Carlos Arciniegas
          </Typography>
          <Typography variant="body2" sx={{ ml:1, color: "text.secondary" }}>
            Frontend, Backend
          </Typography>
        </DashboardBox>
        <DashboardBox gridArea="d" sx={{backgroundColor: "sky.200"}}>
          <Box sx={{ position: "relative", display: "inline-block" }}>
            <Avatar
              variant="rounded"
              imgProps={{
                width: "150",
                height: "150",
                loading: "lazy",
              }}
              alt="Tyler Halfpenny"
              src="/public/icon4.jpeg"
              sx={(theme) => ({
                width: 150,
                height: 150,
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
                transform: "translateX(-100%), translateY(100%)",
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
            sx={{ fontWeight: "bold", ml:1, mt: 2, mb: 0.5 }}
          >
            Tyler Halfpenny
          </Typography>
          <Typography variant="body2" sx={{ ml:1, color: "text.secondary" }}>
            Frontend
          </Typography>
        </DashboardBox>
        <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexWrap: "wrap",
          gap: 1.5,
          width: "100%",
          height: "100%",
          gridTemplateAreas: gridTemplateLargeScreens,
        }}
      >
        <DashboardBox gridArea="d" sx={{backgroundColor: "sky.200"}}>
          <Box sx={{ position: "relative", display: "inline-block" }}>
            <Avatar
              variant="rounded"
              imgProps={{
                width: "150",
                height: "150",
                loading: "lazy",
              }}
              alt="Dr. Rick Leinecker"
              src="/public/icon5.jpeg"
              sx={(theme) => ({
                width: 150,
                height: 150,
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
                transform: "translateX(-100%), translateY(100%)",
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
              aria-label="Dr. Rick Leinecker LinkedIn profile"
              component="a"
              href="https://www.linkedin.com/in/rickleinecker/"
              target="_blank"
              rel="noopener"
            >
              <LinkedInIcon fontSize="small" sx={{ color: "grey.500" }} />
            </IconButton>
          </Box>
          <Typography
            variant="body2"
            sx={{ fontWeight: "bold", ml:1, mt: 2, mb: 0.5 }}
          >
            Dr. Leinecker
          </Typography>
          <Typography variant="body2" sx={{ ml:1, color: "text.secondary" }}>
            Instructor
          </Typography>
        </DashboardBox>
        <DashboardBox gridArea="d" sx={{backgroundColor: "sky.200"}}>
          <Box sx={{ position: "relative", display: "inline-block" }}>
            <Avatar
              variant="rounded"
              imgProps={{
                width: "150",
                height: "150",
                loading: "lazy",
              }}
              alt="Dr. Hwan Choi"
              src="/public/icon6.jpeg"
              sx={(theme) => ({
                width: 150,
                height: 150,
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
                transform: "translateX(-100%), translateY(100%)",
                overflow: "hidden",
              })}
            >
              <img
                loading="lazy"
                height="20"
                width="40"
                src="https://flagcdn.com/kr.svg"
                alt=""
              />
            </Box>
          </Box>
          <Box sx={{ mt: -0.5, mr: -0.5, ml: "auto" }}>
            <IconButton
              aria-label="Dr. Hwan Choi LinkedIn profile"
              component="a"
              href="https://www.linkedin.com/in/hwan-choi-8ba75063/"
              target="_blank"
              rel="noopener"
            >
              <LinkedInIcon fontSize="small" sx={{ color: "grey.500" }} />
            </IconButton>
          </Box>
          <Typography
            variant="body2"
            sx={{ fontWeight: "bold", ml:1, mt: 2, mb: 0.5 }}
          >
            Dr. Choi
          </Typography>
          <Typography variant="body2" sx={{ ml:1, color: "text.secondary" }}>
            Principal Investigator
          </Typography>
        </DashboardBox>
        <DashboardBox gridArea="d" sx={{backgroundColor: "sky.200"}}>
          <Box sx={{ position: "relative", display: "inline-block" }}>
            <Avatar
              variant="rounded"
              imgProps={{
                width: "150",
                height: "150",
                loading: "lazy",
              }}
              alt="Dr. Hossain"
              src="/public/icon7.jpeg"
              sx={(theme) => ({
                width: 150,
                height: 150,
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
                transform: "translateX(-100%), translateY(100%)",
                overflow: "hidden",
              })}
            >
              <img
                loading="lazy"
                height="20"
                width="40"
                src="https://flagcdn.com/bd.svg"
                alt=""
              />
            </Box>
          </Box>
          <Box sx={{ mt: -0.5, mr: -0.5, ml: "auto" }}>
            <IconButton
              aria-label="Dr. Hossain LinkedIn profile"
              component="a"
              href="https://www.linkedin.com/in/hwan-choi-8ba75063/"
              target="_blank"
              rel="noopener"
            >
              <LinkedInIcon fontSize="small" sx={{ color: "grey.500" }} />
            </IconButton>
          </Box>
          <Typography
            variant="body2"
            sx={{ fontWeight: "bold", ml:1, mt: 2, mb: 0.5 }}
          >
            Dr. Hossain
          </Typography>
          <Typography variant="body2" sx={{ ml:1, color: "text.secondary" }}>
            Mentor
          </Typography>
        </DashboardBox>
        <DashboardBox gridArea="d" sx={{backgroundColor: "sky.200"}}>
          <Box sx={{ position: "relative", display: "inline-block" }}>
            <Avatar
              variant="rounded"
              imgProps={{
                width: "150",
                height: "150",
                loading: "lazy",
              }}
              alt="Dr. Hossain"
              src="/public/icon8.jpg"
              sx={(theme) => ({
                width: 150,
                height: 150,
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
                transform: "translateX(-100%), translateY(100%)",
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
              aria-label="LinkedIn profile"
              component="a"
              href="https://www.linkedin.com/"
              target="_blank"
              rel="noopener"
            >
              <LinkedInIcon fontSize="small" sx={{ color: "grey.500" }} />
            </IconButton>
          </Box>
          <Typography
            variant="body2"
            sx={{ fontWeight: "bold", ml:1, mt: 2, mb: 0.5 }}
          >
            Joseph Dranetz
          </Typography>
          <Typography variant="body2" sx={{ ml:1, color: "text.secondary" }}>
            Mentor
          </Typography>
        </DashboardBox>
      </Box>
      
      </Box>
  );
}

export default About;
