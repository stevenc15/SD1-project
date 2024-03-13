import express from 'express'
import dotenv from 'dotenv'
import mongoose from 'mongoose'
import authRouter from './routes/auth_route.js'

const app = express()
dotenv.config()

mongoose.connect(process.env.MONGO_STRING).then(() => {
    console.log('Connected to database')
}).catch((err) => {
    console.log('Fail to connect to database. Error: ', err)
})

app.use(express.json())
app.listen(3000, () => {
    console.log("App listening at port 3000")
})

app.use('/backend/auth', authRouter)

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

