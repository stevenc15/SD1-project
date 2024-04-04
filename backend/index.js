import express from 'express'
import dotenv from 'dotenv'
import bodyParser from 'body-parser'
import mongoose from 'mongoose'
import Grid from 'gridfs-stream'
import authRouter from './routes/auth_route.js'
// import uploadRouter from './routes/upload_route.js'
import multer from 'multer'
import {GridFsStorage} from 'multer-gridfs-storage'
import crypto from 'node:crypto'
import { GridFSBucket } from 'mongodb'
Grid.mongo = mongoose.mongo

const app = express()
dotenv.config()
app.use(bodyParser.json())

// Without this connection, login and sign up endpoints fail.
mongoose.connect(process.env.MONGO_STRING).then(() => {
    console.log('Connected to database')
}).catch((err) => {
    console.log('Fail to connect to database. Error: ', err)
})

const conn = mongoose.createConnection(process.env.MONGO_STRING);
conn.on('connected', () => console.log('Successfully connected to database.'));
conn.on('disconnected', () => console.log('Lost connection with database.'));

// Testing server from Postman
app.get('/', (req, res) => {
  res.send('Server Running')
})

// Initial Grid
let gfs
conn.once('open', () => {
    // Initializing Stream
    gfs = Grid(conn.db);
    gfs.collection('uploads')
})

// Storage Engine
const storage = new GridFsStorage({
    url: process.env.MONGO_STRING,
    file: (req, file) => {
      return new Promise((resolve, reject) => {
        crypto.randomBytes(16, (err, buf) => {
          if (err) {
            return reject('Error: ', err);
          }
          const filename = file.originalname;
          const fileInfo = {
            filename: filename,
            bucketName: 'uploads',
          };
          resolve(fileInfo);
        });
      });
    }
  });

const upload = multer({ storage });

// With multer we can upload an array of files too. For now, this route only uploads a single file
app.post('/upload', upload.single('file'), (req, res) => {
  try {
      res.json({ file: req.file })
      console.log('File uploaded successfully')
  } catch (error) {
      res.send("Not possible to upload file. Error: ", error.message)
  }
})

app.get('/get-files', async (req, res) => {
  try {
      const bucket = new GridFSBucket(conn.db, {bucketName: 'uploads'})
      const files = await bucket.find().toArray()
      if (!files || files.length === 0)
      {
          return res.status(404).json({error: 'No files exists'})
      }
      console.log("Files retrieved successfully")
      return res.json(files)
  } catch (error) {
      res.status(500).json({error: 'Failed to fetch files'})
  }
})

const port = 3000 
app.listen(port, () => {
    console.log("App listening at port", port)
})

app.use('/backend/auth', authRouter)
