import express from 'express'
import dotenv from 'dotenv'
import bodyParser from 'body-parser'
import mongoose from 'mongoose'
import Grid from 'gridfs-stream'
import authRouter from './routes/auth_route.js'
// import uploadRouter from './routes/upload_route.js'
import multer from 'multer'
import {GridFsStorage} from 'multer-gridfs-storage'
import crypto from 'crypto'
import path from 'path'

const app = express()
dotenv.config()

// mongoose.connect(process.env.MONGO_STRING).then(() => {
//     console.log('Connected to database')
// }).catch((err) => {
//     console.log('Fail to connect to database. Error: ', err)
// })

let conn = mongoose.createConnection(process.env.MONGO_STRING);
conn.on('connected', () => console.log('Successfully connected to database.'));
conn.on('disconnected', () => console.log('Lost connection with database.'));

let gfs
conn.once('open', () => {
    // Initializing Stream
    gfs = Grid(conn.db, mongoose.mongo);
    gfs.collection('uploads')
})

const storage = new GridFsStorage({
    url: process.env.MONGO_STRING,
    file: (req, file) => {
      return new Promise((resolve, reject) => {
        crypto.randomBytes(16, (err, buf) => {
          if (err) {
            return reject('Error: ', err);
          }
          const filename = buf.toString('hex') + path.extname(file.originalname);
          const fileInfo = {
            filename: filename,
            bucketName: 'uploads'
          };
          resolve(fileInfo);
        });
      });
    }
  });

const upload = multer({ storage });

app.post('/upload', upload.single('file'), (req, res) => {
    if (!req.file) {
        res.status(413).send(`File not uploaded!`);
        return;
    }
      // successfull completion
    res.status(201).send("Files uploaded successfully");
})

// Middleware
app.use(express.json())
const port = 3000 
app.listen(port, () => {
    console.log("App listening at port", port)
})

app.use('/backend/auth', authRouter)

// Testing server from Postman
app.get('/', (req, res) => {
    res.send('Server Running')
})

// app.use needs this middleware to work.
app.use((err, req, res, next) => {
    const statusCode = err.statusCode || 500 
    const message = err.message || "Internal Server Error" 
    return res.status(statusCode).json({
        success: false,
        statusCode,
        message,
    })
}) 

