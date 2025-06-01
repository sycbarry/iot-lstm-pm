

let GetDataFromAPI = async function() {
    return new Promise((req, rej) => {
        fetch("http://127.0.0.1:5002/consume").then((data) => {
            data.json(raw => { req(raw) })
        })
    })
}

module.exports = GetDataFromAPI